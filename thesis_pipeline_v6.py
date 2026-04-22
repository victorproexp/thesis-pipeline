import os
import re
import random
import subprocess
import plistlib
import unicodedata
import numpy as np
import pandas as pd
import fitz  # PyMuPDF

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Topic modeling
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
from pyvis.network import Network

# ==========================================
# CORPUS CONSTRUCTION: INCLUSION CRITERIA
# ==========================================
# As per methodology section: systematic corpus with explicit inclusion criteria
# to ensure validity and address sampling bias concerns.
#
# INCLUSION CRITERIA:
#   - Time Window: January 2023 – April 2026 (Post-ChatGPT era)
#   - Document Types: Official white papers, strategic reports, policy-oriented
#     blog posts from official company channels
#   - Relevance: Explicit mention of "Education", "Higher Education", 
#     or "Academic" contexts
#   - Source: Official company channels only (blogs, research pages, 
#     policy pages)
#
# SAMPLING STRATEGY:
#   - NO artificial parity enforced: Corpus reflects actual company output
#   - All documents meeting criteria are included
#   - Proportional differences in company output are preserved as data
#   - Weighting in analysis (word count + topic probability) normalizes
#     for natural imbalances
#
# RATIONALE:
#   Small equal samples (5 docs/level) are vulnerable to selection bias.
#   By including all relevant documents and weighting proportionally,
#   we let the data speak to actual communicative scope and emphasis.

INCLUSION_CRITERIA = {
    "time_window": "2023-01-01 to 2026-04-30",
    "document_types": ["white_papers", "strategic_reports", "policy_blog_posts"],
    "relevance_keywords": ["education", "higher education", "academic"],
    "sources": ["official_blogs", "research_pages", "policy_pages"],
    "note": "Inclusion criteria document ensures transparency and defensibility."
}

# ==========================================
# SETUP
# ==========================================
random.seed(42)
np.random.seed(42)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

# HYBRID APPROACH (Statistical Filtering + Minimal Brand Stopwords)
# Combine aggressive min_df/max_df statistical thresholds with targeted brand-name filtering.
# This gives the best of both worlds: automatic noise filtering + explicit brand name removal.

# Minimal safe stopword core:
# Keep only explicit brand/product identifiers and obvious OCR/lemmatization artifacts.
# Preserve concept-bearing discourse terms (e.g., governance, security, design, policy vocabulary)
# so BERTopic can discover meaningful thematic differences naturally.
brand_stopwords = [
    # ========== Brand/Product names (filter branding noise) ==========
    'anthropic', 'microsoft', 'google', 'facebook', 'amazon',
    'azure', 'openai', 'deepmind', 'claude', 'gpt', 'gemini',
    'copilot', 'linkedin', 'palm', 'learnlm', 'llama',

    # ========== Known truncation/OCR artifacts ==========
    'identi', 'classi', 'cogniti', 'speci', 'uency',

    # ========== Domain-generic high-frequency terms ==========
    # These terms dominate many documents and can flatten topic labels.
    'education', 'educational', 'academia', 'academic', 'learning', 
    'teaching', 'student', 'educator', 'principal', 'technology', 
    'university', 'document'
]

# ==========================================
# TEXT PREPROCESSING
# ==========================================
def remove_urls(text: str) -> str:
    """Remove URLs and web references"""
    # Remove http(s) URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove www references
    text = re.sub(r'www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove file paths and domain extensions chains
    text = re.sub(r'([a-zA-Z0-9]+\.)+com\b', '', text)
    text = re.sub(r'([a-zA-Z0-9]+\.)+org\b', '', text)
    text = re.sub(r'([a-zA-Z0-9]+\.)+edu\b', '', text)
    return text


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def lemmatize_text(text: str) -> str:
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(t) for t in tokens])


def preprocess(text: str) -> str:
    text = remove_urls(text)      # Remove URLs first
    text = clean_text(text)
    text = normalize_text(text)
    text = lemmatize_text(text)
    return text


def _decode_finder_comment_xattr(raw_bytes):
    """Decode Finder comment bytes from xattr (binary plist on macOS)."""
    if not raw_bytes:
        return ""

    is_binary_plist = raw_bytes.startswith(b"bplist")

    try:
        parsed = plistlib.loads(raw_bytes)
        if isinstance(parsed, bytes):
            parsed = parsed.decode("utf-8", errors="ignore")
        if isinstance(parsed, str):
            return parsed.strip()
    except Exception:
        if is_binary_plist:
            return ""

    if is_binary_plist:
        return ""

    try:
        return raw_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _decode_finder_comment_xattr_hex(raw_hex_bytes):
    """Decode Finder comment when xattr returns hex-encoded output."""
    if not raw_hex_bytes:
        return ""

    try:
        hex_text = raw_hex_bytes.decode("utf-8", errors="ignore")
        hex_text = re.sub(r"\s+", "", hex_text)
        if not hex_text:
            return ""
        decoded = bytes.fromhex(hex_text)
        return _decode_finder_comment_xattr(decoded)
    except Exception:
        return ""


def read_finder_comment(filepath):
    """Read Finder comment with mdls first, then xattr fallback for moved files."""
    candidates = []
    for candidate in (filepath, os.path.abspath(filepath), os.path.realpath(filepath)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    normalized_candidates = []
    for candidate in candidates:
        for form in ("NFC", "NFD"):
            normalized = unicodedata.normalize(form, candidate)
            if normalized not in normalized_candidates:
                normalized_candidates.append(normalized)

    candidates = normalized_candidates

    for candidate in candidates:
        try:
            result = subprocess.run(
                ["mdls", "-name", "kMDItemFinderComment", "-raw", candidate],
                capture_output=True, text=True, timeout=15
            )
            comment = result.stdout.strip()
            if comment and comment != "(null)":
                return comment
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["mdls", "-name", "kMDItemFinderComment", candidate],
                capture_output=True, text=True, timeout=15
            )
            raw_out = (result.stdout or "").strip()
            match = re.search(r'kMDItemFinderComment\s*=\s*"(.*)"$', raw_out)
            if match:
                comment = match.group(1).strip()
                if comment:
                    return comment
        except Exception:
            pass

        # Some moved files keep Finder comments in xattr before Spotlight refreshes.
        try:
            xattr_result = subprocess.run(
                ["xattr", "-p", "com.apple.metadata:kMDItemFinderComment", candidate],
                capture_output=True, timeout=15
            )
            if xattr_result.returncode == 0 and xattr_result.stdout:
                comment = _decode_finder_comment_xattr(xattr_result.stdout)
                if comment and comment != "(null)":
                    return comment
        except Exception:
            pass

        try:
            xattr_hex_result = subprocess.run(
                ["xattr", "-px", "com.apple.metadata:kMDItemFinderComment", candidate],
                capture_output=True, timeout=15
            )
            if xattr_hex_result.returncode == 0 and xattr_hex_result.stdout:
                comment = _decode_finder_comment_xattr_hex(xattr_hex_result.stdout)
                if comment and comment != "(null)":
                    return comment
        except Exception:
            continue

    return ""


# ==========================================
# PDF MINING
# ==========================================
def mine_pdfs(base_dir):
    """
    Extract PDFs WITHOUT chunking - preserve full document context.
    
    CHANGE: Reverted to full-document processing to avoid "echo chamber" topics.
    Instead, word count weighting in Knowledge Graph handles proportional representation.
    BERTopic probability distribution enables soft clustering (multi-topic mapping).
    """
    data = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".pdf"):
                continue

            path = os.path.join(root, file)

            path_parts = root.split(os.sep)
            level = path_parts[-2] if len(path_parts) > 2 else "Unknown_Level"
            company = path_parts[-1] if len(path_parts) > 1 else "Unknown_Company"

            try:
                doc = fitz.open(path)
                text = ""

                for page in doc:
                    text += page.get_text("text") + " "

                doc.close()

                cleaned = clean_text(text)
                processed = preprocess(cleaned)
                
                # NO chunking - preserved full documents for meaningful topic modeling
                data.append({
                    "Level": level,
                    "Company": company,
                    "FileName": file,
                    "Raw_Text": cleaned,
                    "Processed_Text": processed,
                    "Word_Count": len(cleaned.split())
                })

                print(f"[OK] {company} -> {file}")
            except Exception as e:
                print(f"[ERROR] {file}: {e}")

    return pd.DataFrame(data)


# ==========================================
# TOKENIZER FOR VECTORIZER
# ==========================================
def custom_tokenizer(text):
    return [
        word.lower() for word in text.split()
        if len(word) > 2  # Simple length filter only
    ]


# ==========================================
# POST-PROCESSING: TOPIC CLEANING
# ==========================================
def clean_topic_words(words_str: str, max_words: int = 10) -> str:
    """
    Post-process topic words from CSV;
    1. Parse the words list
    2. Split multi-word phrases into individual words
    3. Remove duplicate/redundant prefixes
    4. Return cleaned comma-separated string
    """
    import ast
    
    try:
        words = ast.literal_eval(words_str)
    except:
        # Fallback parsing
        words = [w.strip().strip("'") for w in words_str.strip("[]").split(",")]
    
    cleaned = []
    seen_prefixes = set()
    
    for phrase in words:
        if not isinstance(phrase, str):
            continue
            
        # Split multi-word phrases
        phrase_words = phrase.split()
        
        for word in phrase_words:
            # Skip if word is too short or empty
            if len(word) <= 2 or not word.isalpha():
                continue
            
            word = word.lower().strip()
            
            # Skip common redundant prefixes if already seen
            if word in {'education', 'learning', 'teaching', 'student', 'educator', 'initiative'}:
                if word in seen_prefixes:
                    continue
                seen_prefixes.add(word)
            
            # Skip if already in cleaned list
            if word not in cleaned:
                cleaned.append(word)
    
    return str(cleaned[:max_words])  # Return as string representation


def postprocess_topics_csv(csv_path: str):
    """
    Post-process saved topics.csv to:
    1. Remove any stopwords that re-appeared in Representation
    2. Clean word representations
    3. Ensure only clean, semantically meaningful words remain
    """
    import pandas as pd
    import ast
    
    df = pd.read_csv(csv_path, dtype={'Representation': str})
    
    # Filter function: remove stopwords from representation
    def remove_stopwords_from_rep(rep_str: str, stopwords_set: set) -> str:
        try:
            words = ast.literal_eval(rep_str)
            if not isinstance(words, list):
                return rep_str
            
            # Filter out stopwords and keep only the remaining words
            filtered = [w for w in words if isinstance(w, str) and w.lower() not in stopwords_set]
            # Also remove duplicates while preserving order
            seen = set()
            unique_filtered = []
            for w in filtered:
                w_lower = w.lower()
                if w_lower not in seen:
                    unique_filtered.append(w)
                    seen.add(w_lower)
            
            return str(unique_filtered[:10])  # Keep top 10
        except:
            return rep_str
    
    # Create stopwords set (include all stopwords for filtering)
    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = nltk_stopwords.union(set(brand_stopwords))
    
    # Apply stopword removal to Representation column
    df['Representation'] = df['Representation'].apply(
        lambda x: remove_stopwords_from_rep(x, all_stopwords)
    )
    
    # Apply additional cleaning
    df['Representation'] = df['Representation'].apply(
        lambda x: clean_topic_words(x, max_words=10)
    )
    
    # Save cleaned version
    df.to_csv(csv_path, index=False)
    print(f"[OK] Topics cleaned (stopwords removed) and saved to {csv_path}")


def save_cluster_diagnostics(corpus_with_topics_path: str, topics_csv_path: str, output_path: str):
    """
    Compute and save BERTopic cluster diagnostics for reproducible model quality tracking.
    """
    import ast

    corpus_df = pd.read_csv(corpus_with_topics_path)
    topics_df = pd.read_csv(topics_csv_path)

    doc_count = len(corpus_df)
    noise_docs = int((corpus_df["Topic"] == -1).sum())
    noise_ratio = float(noise_docs / doc_count) if doc_count else 0.0

    non_noise_df = corpus_df[corpus_df["Topic"] != -1].copy()
    num_topics_excl_noise = int(topics_df[topics_df["Topic"] != -1]["Topic"].nunique())

    # Topic size balance: normalized entropy in [0, 1]
    counts = non_noise_df["Topic"].value_counts().sort_index()
    if len(counts) > 0 and counts.sum() > 0:
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        max_entropy = float(np.log(len(p))) if len(p) > 1 else 1.0
        topic_balance_entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        topic_balance_entropy_norm = 0.0

    # Topic overlap from top words in Representation (lower is better)
    representation_values = topics_df[topics_df["Topic"] != -1]["Representation"].tolist()
    word_sets = []
    for rep in representation_values:
        try:
            words = ast.literal_eval(str(rep))
            if isinstance(words, list):
                word_sets.append(set(str(w).lower() for w in words[:10]))
            else:
                word_sets.append(set())
        except Exception:
            word_sets.append(set())

    jaccards = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            a, b = word_sets[i], word_sets[j]
            if a or b:
                jaccards.append(len(a & b) / max(1, len(a | b)))
    mean_topic_word_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

    # Confidence: mean of max topic probability per document
    def _parse_probs(raw_value):
        arr = np.fromstring(str(raw_value).strip("[]"), sep=" ")
        return arr if arr.size else np.array([])

    max_probs = []
    for raw_probs in corpus_df["Topic_Distribution"].dropna().tolist():
        arr = _parse_probs(raw_probs)
        if arr.size:
            max_probs.append(float(arr.max()))
    mean_max_topic_probability = float(np.mean(max_probs)) if max_probs else np.nan

    # Separation diagnostics in TF-IDF space (fast and stable)
    silhouette_cosine_non_noise = np.nan
    davies_bouldin_non_noise = np.nan
    if len(non_noise_df) > 10 and non_noise_df["Topic"].nunique() > 1:
        valid_text_mask = (
            non_noise_df["Processed_Text"].notna()
            & non_noise_df["Processed_Text"].astype(str).str.strip().astype(bool)
        )
        valid_texts = non_noise_df.loc[valid_text_mask, "Processed_Text"].astype(str).tolist()
        valid_labels = non_noise_df.loc[valid_text_mask, "Topic"].to_numpy()
        invalid_text_docs = int((~valid_text_mask).sum())

        if invalid_text_docs > 0:
            print(f"[WARN] Skipping {invalid_text_docs} non-noise documents with missing/empty Processed_Text for TF-IDF diagnostics")

        if len(valid_texts) > 10 and len(np.unique(valid_labels)) > 1:
            vectorizer = TfidfVectorizer(
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2),
                max_features=10000,
            )
            X = vectorizer.fit_transform(valid_texts)
            silhouette_cosine_non_noise = float(silhouette_score(X, valid_labels, metric="cosine"))
            davies_bouldin_non_noise = float(davies_bouldin_score(X.toarray(), valid_labels))

    diagnostics_df = pd.DataFrame([
        {
            "doc_count": doc_count,
            "num_topics_excl_noise": num_topics_excl_noise,
            "noise_docs": noise_docs,
            "noise_ratio": noise_ratio,
            "topic_balance_entropy_norm": topic_balance_entropy_norm,
            "mean_topic_word_jaccard": mean_topic_word_jaccard,
            "mean_max_topic_probability": mean_max_topic_probability,
            "silhouette_cosine_non_noise": silhouette_cosine_non_noise,
            "davies_bouldin_non_noise": davies_bouldin_non_noise,
        }
    ])

    diagnostics_df.to_csv(output_path, index=False)
    print(f"[OK] Cluster diagnostics saved: {output_path}")
    print(
        "    "
        f"topics={num_topics_excl_noise}, noise={noise_docs}/{doc_count} ({noise_ratio:.1%}), "
        f"silhouette={silhouette_cosine_non_noise:.3f}, "
        f"max_prob={mean_max_topic_probability:.3f}"
    )


# ==========================================
# KNOWLEDGE GRAPH
# ==========================================
def generate_knowledge_graph(df, topic_model, output_path):
    """
    Generate knowledge graph with AGGREGATED company→topic edges.
    
    Methodology:
      1. Aggregate per-document probabilities to company→topic level
      2. Edge weight encodes: avg_probability × company_discourse_share
      3. Company-colored edges for immediate visual traceability
      4. Topic node size ∝ document count; company node size ∝ word volume
      5. Values terms highlighted on topic nodes as secondary label
      6. Embedded legend explains all visual encodings
    """
    # Values vocabulary for overlay on topic labels
    VALUES_TERMS = {
        'governance', 'transparency', 'accountability', 'autonomy',
        'privacy', 'security', 'equity', 'oversight', 'democratic',
        'agency', 'ethical', 'responsible', 'public', 'openness',
        'regulation', 'infrastructure', 'safety', 'fairness',
        'inclusion', 'access', 'trust', 'wellbeing', 'diversity',
        'participation',
    }
    COMPANY_COLORS = {
        "Google":    {"node": "#4285F4", "edge": "#4285F4"},  # Google blue
        "Microsoft": {"node": "#00A4EF", "edge": "#00A4EF"},  # Microsoft blue
        "Anthropic": {"node": "#D97757", "edge": "#D97757"},  # Anthropic orange
    }
    TOPIC_COLOR = "#C4A7E7"   # Soft purple for topic nodes
    BG_COLOR = "#1a1a2e"      # Deep navy background

    net = Network(height="800px", width="100%", bgcolor=BG_COLOR, font_color="white")

    # ---- Company nodes (size ∝ discourse volume) ----
    company_totals = {}
    for company in df["Company"].unique():
        if company == "Unknown_Company":
            continue
        company_df = df[df["Company"] == company]
        total_words = company_df["Word_Count"].sum() if "Word_Count" in company_df.columns else sum(
            len(str(t).split()) for t in company_df["Raw_Text"]
        )
        company_totals[company] = total_words

    # Scale company node sizes: 30–55 range based on word volume
    if company_totals:
        max_vol = max(company_totals.values())
        min_vol = min(company_totals.values())
        vol_range = max_vol - min_vol if max_vol != min_vol else 1

    for company, total_words in company_totals.items():
        size = 30 + 25 * (total_words - min_vol) / vol_range
        colors = COMPANY_COLORS.get(company, {"node": "#FFD700"})
        doc_count = len(df[df["Company"] == company])
        net.add_node(
            company,
            label=company,
            color=colors["node"],
            size=size,
            font={"size": 18, "color": "white", "bold": True},
            title=f"{company}\n{doc_count} documents\n{total_words:,} words"
        )

    # ---- Topic nodes (size ∝ document count) ----
    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info.Topic != -1]

    for _, trow in valid.iterrows():
        topic_id = trow["Topic"]
        doc_count = trow["Count"]
        words = topic_model.get_topic(topic_id)[:8]  # Grab extra to survive dedup
        # Deduplicate: skip unigrams already contained in a displayed bigram
        seen_parts = set()
        deduped = []
        for w, s in words:
            if ' ' in w:  # Bigram — register its parts
                seen_parts.update(w.split())
                deduped.append((w, s))
            elif w not in seen_parts:
                deduped.append((w, s))
            if len(deduped) == 5:
                break

        # Identify values terms in the full top-15 keywords
        all_words = topic_model.get_topic(topic_id)[:15]
        values_in_topic = []
        for w, s in all_words:
            # Check each part of bigrams and unigrams against values vocabulary
            parts = w.split() if ' ' in w else [w]
            for p in parts:
                if p in VALUES_TERMS and p not in values_in_topic:
                    values_in_topic.append(p)

        keyword_line = " · ".join([w for w, _ in deduped])
        values_line = f"⟨{', '.join(values_in_topic)}⟩" if values_in_topic else ""
        label = f"T{topic_id}\n{keyword_line}"
        if values_line:
            label += f"\n{values_line}"

        size = 15 + doc_count * 1.2  # Scale with doc count
        net.add_node(
            int(topic_id),
            label=label,
            color=TOPIC_COLOR,
            size=size,
            font={"size": 13, "color": "white", "multi": True},
            title=f"Topic {topic_id} ({doc_count} docs)\n" +
                  "\n".join([f"  {w} ({s:.3f})" for w, s in words]) +
                  (f"\n\nValues terms: {', '.join(values_in_topic)}" if values_in_topic else "\n\nNo values terms in top keywords")
        )

    # ---- Aggregate company→topic edges ----
    # Instead of per-document edges, compute one edge per (company, topic) pair:
    #   weight = avg_probability × discourse_share_fraction × scale
    total_corpus_words = sum(company_totals.values()) if company_totals else 1

    edge_data = {}  # (company, topic_id) → {prob_sum, doc_count, word_sum}
    for _, row in df.iterrows():
        company = str(row["Company"])
        if company == "Unknown_Company":
            continue
        word_count = int(row["Word_Count"]) if "Word_Count" in row else len(str(row["Processed_Text"]).split())
        probs = row["Topic_Distribution"]
        for topic_id, prob in enumerate(probs):
            key = (company, topic_id)
            if key not in edge_data:
                edge_data[key] = {"prob_sum": 0.0, "doc_count": 0, "word_sum": 0}
            edge_data[key]["prob_sum"] += prob
            edge_data[key]["doc_count"] += 1
            edge_data[key]["word_sum"] += word_count

    threshold_prob = 0.15  # Show edges where avg probability ≥ 15%

    for (company, topic_id), stats in edge_data.items():
        avg_prob = stats["prob_sum"] / stats["doc_count"]
        if avg_prob < threshold_prob:
            continue
        discourse_share = stats["word_sum"] / total_corpus_words
        edge_weight = avg_prob * discourse_share * 300  # Scaled for visual thickness

        colors = COMPANY_COLORS.get(company, {"edge": "#888888"})
        net.add_edge(
            company,
            int(topic_id),
            value=float(edge_weight),
            color=colors["edge"],
            title=f"{company} → Topic {topic_id}\n"
                  f"Avg. probability: {avg_prob:.0%}\n"
                  f"Documents: {stats['doc_count']}\n"
                  f"Discourse share: {discourse_share:.1%}"
        )

    # ---- Physics layout ----
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -250,
          "centralGravity": 0.008,
          "springLength": 350,
          "springConstant": 0.04,
          "damping": 0.5,
          "avoidOverlap": 0.8
        },
        "stabilization": {"iterations": 400, "fit": true}
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "scaling": {"min": 1, "max": 18}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragNodes": true,
        "dragView": true
      }
    }
    """)

    # Write base HTML, then inject legend overlay
    net.write_html(output_path)

    # ---- Inject legend + title into the HTML ----
    legend_html = """
    <div style="position:absolute;top:12px;left:12px;background:rgba(26,26,46,0.92);
                padding:14px 18px;border-radius:8px;font-family:system-ui,sans-serif;
                color:#ccc;font-size:12px;line-height:1.7;z-index:10;
                border:1px solid rgba(255,255,255,0.1);">
      <div style="font-size:15px;font-weight:700;color:#fff;margin-bottom:6px;">
        Company–Topic Knowledge Graph
      </div>
      <div><span style="color:#4285F4;">●</span> Google &nbsp;
           <span style="color:#00A4EF;">●</span> Microsoft &nbsp;
           <span style="color:#D97757;">●</span> Anthropic &nbsp;
           <span style="color:#C4A7E7;">●</span> Topic</div>
      <div style="margin-top:6px;border-top:1px solid rgba(255,255,255,0.15);padding-top:6px;">
        Node size = discourse volume / doc count<br>
        Edge thickness = avg. probability × discourse share<br>
        Edge color = company affiliation<br>
        ⟨ ⟩ = democratic public values terms (Marginson)
      </div>
    </div>
    """
    with open(output_path, "r") as f:
        html = f.read()
    # JS: disable physics after stabilization so nodes can be dragged freely
    freeze_js = """
    <script type="text/javascript">
      // After the network stabilizes, turn off physics so dragging is instant
      document.addEventListener("DOMContentLoaded", function() {
        var checkNetwork = setInterval(function() {
          if (typeof network !== 'undefined') {
            clearInterval(checkNetwork);
            network.on("stabilized", function() {
              network.setOptions({physics: {enabled: false}});
            });
          }
        }, 100);
      });
    </script>
    """
    html = html.replace('<div id="mynetwork"', legend_html + '\n        <div id="mynetwork"')
    html = html.replace('</body>', freeze_js + '\n</body>')
    with open(output_path, "w") as f:
        f.write(html)


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    BASE_DIR = "./Thesis_Data_Mining"
    OUTPUT_DIR = os.path.join(BASE_DIR, "04_Analysis_Outputs")
    EXCLUDED_DIR = "./Excluded_Documents_Low_Relevance"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Methodology enforcement: exclude documents older than 2023 before analysis.
    STRICT_YEAR_FILTER = True

    def get_finder_comment(filepath):
        return read_finder_comment(filepath)

    def parse_url_and_year_from_comment(comment):
        if not comment:
            return "", ""

        cleaned = str(comment).strip().strip('"').strip("'")
        cleaned = re.sub(r"\s+", " ", cleaned)

        url_match = re.search(r"https?://\S+", cleaned)
        url = url_match.group(0).rstrip(".,;)\"]") if url_match else ""

        year_match = re.search(r"(20\d{2})(?!.*20\d{2})", cleaned)
        year = year_match.group(1) if year_match else ""
        if year and not (2000 <= int(year) <= 2026):
            year = ""

        return url, year

    def infer_date_evidence_from_comment(comment, publication_year):
        """Classify Finder comment evidence quality for audit transparency."""
        if publication_year:
            return "finder_comment"
        return "finder_comment_no_year" if str(comment).strip() else "missing_finder_comment"

    # 1. Data extraction
    year_excluded_df = pd.DataFrame(columns=[
        "Company", "Level", "FileName", "PublicationYear", "DateEvidence", "URL", "ExclusionReason"
    ])
    df = mine_pdfs(BASE_DIR)

    if df.empty:
        print("No data found.")
        exit()

    # Enforce year-based inclusion rule for the analytical corpus.
    if STRICT_YEAR_FILTER:
        pre_filter_count = len(df)
        comments = df.apply(
            lambda row: get_finder_comment(
                os.path.join(BASE_DIR, row['Level'], row['Company'], row['FileName'])
            ),
            axis=1
        )
        parsed_meta = comments.apply(parse_url_and_year_from_comment)
        df['CommentURL'] = parsed_meta.apply(lambda x: x[0])
        df['PublicationYear'] = parsed_meta.apply(lambda x: x[1])
        df['DateEvidence'] = df.apply(
            lambda row: infer_date_evidence_from_comment(comments.loc[row.name], row['PublicationYear']),
            axis=1
        )
        excluded_mask = ~df['PublicationYear'].apply(lambda y: str(y).isdigit() and int(y) >= 2023)
        excluded_df = df[excluded_mask][
            ['Company', 'Level', 'FileName', 'PublicationYear', 'DateEvidence', 'CommentURL']
        ].copy()
        if not excluded_df.empty:
            excluded_df = excluded_df.rename(columns={'CommentURL': 'URL'})
            excluded_df['ExclusionReason'] = "PublicationYear before 2023 or missing"
            year_excluded_df = excluded_df.copy()
            excluded_path = os.path.join(OUTPUT_DIR, "corpus_excluded_by_year.csv")
            excluded_df.to_csv(excluded_path, index=False)
            print(f"[OK] Excluded docs log saved: {excluded_path}")
            print(f"    Excluded by year rule: {len(excluded_df)}")

        df = df[~excluded_mask].copy().reset_index(drop=True)
        df = df.drop(columns=['PublicationYear', 'CommentURL', 'DateEvidence'], errors='ignore')
        print(f"[OK] Year filter applied: {pre_filter_count} -> {len(df)} documents")

        if df.empty:
            print("No documents remain after 2023+ year filter.")
            exit()

    df.to_csv(os.path.join(OUTPUT_DIR, "corpus.csv"), index=False)

    docs = df["Processed_Text"].tolist()

    # ------------------------------------------
    # BERTopic with Hybrid Filtering
    # ------------------------------------------
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    umap_model = UMAP(
        n_neighbors=12,        # Tuned: reduces over-fragmentation while preserving separation
        n_components=5,
        min_dist=0.30,         # Tuned: smoother manifold for more stable topic grouping
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=4,    # Tuned: controls tiny clusters while keeping noise low
        min_samples=1,         # Relaxed from 2 → minimises noise assignment for borderline docs
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # HYBRID APPROACH: Statistical Filtering + Minimal Brand Stopwords
    # - min_df=2: Word must appear in at least 2 documents (broader corpus, less restrictive)
    # - max_df=0.90: Word appears in at most 90% of documents (allows broader terms)
    # - brand_stopwords: Explicit removal of company/product names that statistically persist
    # 
    # This is more intelligent than pure statistical filtering and requires less manual curation
    # than comprehensive stopword lists.
    
    nltk_stopwords = set(stopwords.words('english'))
    combined_stopwords = list(nltk_stopwords.union(set(brand_stopwords)))
    
    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=(1, 2),
        min_df=2,             # Word must appear in ≥2 documents
        max_df=1.0,           # Disabled: stopwords handle high-freq noise; avoids BERTopic c-TF-IDF crash
        stop_words=combined_stopwords  # NLTK + minimal brand names
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        min_topic_size=5,      # Tuned: favors broader, interpretable topics over fragmented ones
        top_n_words=10,        # Explicitly set top words to extract
        verbose=True
    )

    # ------------------------------------------
    # Train model
    # ------------------------------------------
    topics, probs = topic_model.fit_transform(docs)

    df["Topic"] = topics  # Assign topic ID to each document
    df["Topic_Distribution"] = list(probs)

    # Save corpus with topic assignments (so you can see which docs are noise)
    df.to_csv(os.path.join(OUTPUT_DIR, "corpus_with_topics.csv"), index=False)

    # Save topics
    topic_info = topic_model.get_topic_info()
    
    # IMPROVEMENT: Drop Representative_Docs since Representation column has the key info
    # This keeps the CSV clean and AI-friendly without the large document excerpts
    if 'Representative_Docs' in topic_info.columns:
        topic_info = topic_info.drop(columns=['Representative_Docs'])
    
    topics_csv_path = os.path.join(OUTPUT_DIR, "topics.csv")
    topic_info.to_csv(topics_csv_path, index=False)

    # ------------------------------------------
    # Post-process topics (clean words in CSV)
    # ------------------------------------------
    postprocess_topics_csv(topics_csv_path)

    # ------------------------------------------
    # Cluster diagnostics (automatic quality log)
    # ------------------------------------------
    save_cluster_diagnostics(
        corpus_with_topics_path=os.path.join(OUTPUT_DIR, "corpus_with_topics.csv"),
        topics_csv_path=topics_csv_path,
        output_path=os.path.join(OUTPUT_DIR, "cluster_diagnostics.csv")
    )

    print("\nTop topics:")
    print(topic_info.head(10))

    # ------------------------------------------
    # Identify Noise Documents
    # ------------------------------------------
    noise_docs = df[df["Topic"] == -1][["Company", "FileName", "Topic"]]
    if len(noise_docs) > 0:
        print(f"\n[NOISE] {len(noise_docs)} documents assigned to noise topic (-1):")
        print(noise_docs.to_string(index=False))
    else:
        print("\n[NOISE] No documents assigned to noise topic!")

    # ------------------------------------------
    # Visualizations
    # ------------------------------------------
    try:
        fig = topic_model.visualize_barchart(top_n_topics=10)
        fig.write_html(os.path.join(OUTPUT_DIR, "barchart_overall_topics.html"))
        print(f"[OK] Saved: barchart_overall_topics.html")
    except Exception as e:
        print(f"[ERROR] Could not generate overall barchart: {e}")
    
    # Topics by Hierarchy Level
    try:
        topics_per_level = topic_model.topics_per_class(docs, classes=df["Level"].tolist())
        fig_level = topic_model.visualize_topics_per_class(topics_per_level, top_n_topics=10)
        fig_level.write_html(os.path.join(OUTPUT_DIR, "topics_by_hierarchy_level.html"))
        print(f"[OK] Saved: topics_by_hierarchy_level.html")
    except Exception as e:
        print(f"[ERROR] Could not generate level barchart: {e}")
    
    # Topics by Company
    try:
        topics_per_company = topic_model.topics_per_class(docs, classes=df["Company"].tolist())
        fig_company = topic_model.visualize_topics_per_class(topics_per_company, top_n_topics=10)
        fig_company.write_html(os.path.join(OUTPUT_DIR, "topics_by_company.html"))
        print(f"[OK] Saved: topics_by_company.html")
    except Exception as e:
        print(f"[ERROR] Could not generate company barchart: {e}")

    # ------------------------------------------
    # Knowledge graph
    # ------------------------------------------
    generate_knowledge_graph(
        df,
        topic_model,
        os.path.join(OUTPUT_DIR, "knowledge_graph.html")
    )

    # ------------------------------------------
    # Generate Appendix: Corpus Metadata
    # ------------------------------------------
    # As per supervisor feedback: transparency in sampling process via
    # comprehensive metadata table listing all documents.
    # This table should appear in thesis appendix.
    
    def get_finder_comment(filepath):
        """Read Finder comment via robust mdls+xattr lookup."""
        return read_finder_comment(filepath)

    def parse_url_and_year_from_comment(comment):
        """Parse Finder comment and return best-effort (url, year)."""
        if not comment:
            return "", ""

        cleaned = str(comment).strip().strip('"').strip("'")
        cleaned = re.sub(r"\s+", " ", cleaned)

        url_match = re.search(r"https?://\S+", cleaned)
        url = url_match.group(0).rstrip(".,;)\"]") if url_match else ""

        year_match = re.search(r"(20\d{2})(?!.*20\d{2})", cleaned)
        year = year_match.group(1) if year_match else ""
        if year and not (2000 <= int(year) <= 2026):
            year = ""

        return url, year

    def classify_document_type(filename, url):
        """Classify document type for appendix transparency."""
        fname = str(filename).lower()
        link = str(url).lower()

        if "youtube.com" in link or "transcript" in fname:
            return "Transcript"
        if "blog." in link or "/blog/" in link or "/news/" in link:
            return "Blog Post"
        if "arxiv.org" in link or re.match(r"^\d{4}\.\d{5}", str(filename)):
            return "Research Paper"
        if "whitepaper" in fname or "white-paper" in fname:
            return "White Paper"
        if "report" in fname:
            return "Report"
        if "policy" in fname:
            return "Policy Document"
        if "guide" in fname or "toolkit" in fname or "ebook" in fname or "e-book" in fname:
            return "Guide/Toolkit"
        return "Document"

    def extract_publication_year_and_evidence(filepath, url, comment):
        """Extract publication year and source evidence.
        Returns tuple: (year, evidence_source)
        """
        # Finder comment is the single source of truth for year.
        _, year_from_comment = parse_url_and_year_from_comment(comment)
        if year_from_comment:
            return year_from_comment, "finder_comment"

        return "", "missing"

    appendix_df = df[['Company', 'Level', 'FileName']].copy()
    appendix_df['WordCount'] = df['Raw_Text'].apply(lambda x: len(str(x).split()))

    # Extract Finder comment (format: "URL YYYY")
    full_comments = df.apply(
        lambda row: get_finder_comment(
            os.path.join(BASE_DIR, row['Level'], row['Company'], row['FileName'])
        ),
        axis=1
    )
    
    # URL is sourced strictly from Finder comments.
    comment_urls = full_comments.apply(lambda c: parse_url_and_year_from_comment(c)[0])
    appendix_df['URL'] = comment_urls
    year_evidence = df.apply(
        lambda row: extract_publication_year_and_evidence(
            os.path.join(BASE_DIR, row['Level'], row['Company'], row['FileName']),
            appendix_df.loc[row.name, 'URL'],
            full_comments.loc[row.name]
        ),
        axis=1
    )
    appendix_df['PublicationYear'] = year_evidence.apply(lambda x: x[0])
    appendix_df['DateEvidence'] = year_evidence.apply(lambda x: x[1])
    appendix_df['DocumentType'] = appendix_df.apply(
        lambda row: classify_document_type(row['FileName'], row['URL']),
        axis=1
    )

    # Explicit 2023+ inclusion validation (do not silently drop rows).
    appendix_df['IncludedByYearRule'] = appendix_df['PublicationYear'].apply(
        lambda y: "Yes" if str(y).isdigit() and int(y) >= 2023 else "No"
    )
    appendix_df['ExclusionReason'] = appendix_df.apply(
        lambda row: "" if row['IncludedByYearRule'] == "Yes" else "PublicationYear before 2023 or missing",
        axis=1
    )
    
    # Save defensible sampling frame log (all included rows + rule checks).
    selection_log_df = appendix_df.copy()
    selection_log_df['FilePath'] = selection_log_df.apply(
        lambda row: f"Thesis_Data_Mining/{row['Level']}/{row['Company']}/{row['FileName']}",
        axis=1
    )
    selection_log_df = selection_log_df[[
        'Company', 'Level', 'FileName', 'FilePath', 'DocumentType', 'PublicationYear',
        'DateEvidence', 'IncludedByYearRule', 'ExclusionReason', 'URL'
    ]]

    # Reorder appendix columns (reader-facing appendix table).
    appendix_df = appendix_df[[
        'Company', 'Level', 'FileName', 'DocumentType', 'PublicationYear',
        'WordCount', 'URL'
    ]]

    # Document exclusions placed in Excluded_Documents_Low_Relevance.
    def infer_company_from_name(filename):
        name = str(filename).lower()
        if 'google' in name:
            return 'Google'
        if 'microsoft' in name:
            return 'Microsoft'
        if 'anthropic' in name:
            return 'Anthropic'
        return 'Unknown_Company'

    excluded_low_rel_rows = []
    if os.path.isdir(EXCLUDED_DIR):
        for root, _, files in os.walk(EXCLUDED_DIR):
            for file in files:
                if not file.lower().endswith('.pdf'):
                    continue

                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, ".")
                comment = get_finder_comment(path)
                url_from_comment, _ = parse_url_and_year_from_comment(comment)
                pub_year, evidence = extract_publication_year_and_evidence(path, url_from_comment, comment)

                path_parts = os.path.normpath(rel_path).split(os.sep)
                level = "Unknown_Level"
                company = infer_company_from_name(file)
                if len(path_parts) >= 4:
                    # Expected nested format: Excluded_Documents_Low_Relevance/Level/Company/file.pdf
                    level = path_parts[-3]
                    company = path_parts[-2]

                excluded_low_rel_rows.append({
                    'Company': company,
                    'Level': level,
                    'FileName': file,
                    'FilePath': rel_path,
                    'DocumentType': classify_document_type(file, url_from_comment),
                    'PublicationYear': pub_year,
                    'DateEvidence': evidence,
                    'IncludedByYearRule': "No",
                    'ExclusionReason': "Low relevance (manual exclusion folder)",
                    'URL': url_from_comment,
                })

    excluded_low_rel_df = pd.DataFrame(excluded_low_rel_rows)

    # Build a single audit trail across included and excluded documents.
    audit_frames = [selection_log_df.copy()]
    if not year_excluded_df.empty:
        year_excl = year_excluded_df.copy()
        year_excl['FilePath'] = year_excl.apply(
            lambda row: f"Thesis_Data_Mining/{row['Level']}/{row['Company']}/{row['FileName']}",
            axis=1
        )
        year_excl['DocumentType'] = year_excl['FileName'].apply(lambda n: classify_document_type(n, ""))
        year_excl['IncludedByYearRule'] = "No"
        year_excl = year_excl[[
            'Company', 'Level', 'FileName', 'FilePath', 'DocumentType', 'PublicationYear',
            'DateEvidence', 'IncludedByYearRule', 'ExclusionReason', 'URL'
        ]]
        audit_frames.append(year_excl)
    if not excluded_low_rel_df.empty:
        audit_frames.append(excluded_low_rel_df)

    corpus_audit_df = pd.concat(audit_frames, ignore_index=True)
    
    appendix_csv_path = os.path.join(OUTPUT_DIR, "appendix_corpus_metadata.csv")
    appendix_df.to_csv(appendix_csv_path, index=False)
    selection_log_path = os.path.join(OUTPUT_DIR, "corpus_selection_log.csv")
    selection_log_df.to_csv(selection_log_path, index=False)
    excluded_low_rel_path = os.path.join(OUTPUT_DIR, "corpus_excluded_low_relevance.csv")
    excluded_low_rel_df.to_csv(excluded_low_rel_path, index=False)
    audit_path = os.path.join(OUTPUT_DIR, "corpus_selection_audit.csv")
    corpus_audit_df.to_csv(audit_path, index=False)
    print(f"[OK] Appendix metadata saved: {appendix_csv_path}")
    print(f"[OK] Selection log saved: {selection_log_path}")
    print(f"[OK] Low-relevance exclusions saved: {excluded_low_rel_path}")
    print(f"[OK] Unified corpus audit saved: {audit_path}")
    print(f"    Total documents: {len(appendix_df)}")
    excluded_by_year = len(year_excluded_df)
    print(f"    Excluded by 2023+ year rule: {excluded_by_year}")
    print(f"    Excluded in low-relevance folder: {len(excluded_low_rel_df)}")
    print(f"    Documents per company:")
    for company in appendix_df['Company'].unique():
        if company != "Unknown_Company":
            count = len(appendix_df[appendix_df['Company'] == company])
            total_words = appendix_df[appendix_df['Company'] == company]['WordCount'].sum()
            print(f"      - {company}: {count} documents, {total_words:,} words")

    # ------------------------------------------
    # VALUES TERM FREQUENCY EXPORT (per-company)
    # ------------------------------------------
    # Expanded vocabulary: Marginson democratic public values + van Dijck platform terms
    from collections import Counter

    values_terms = [
        # Governance & oversight (Marginson / Mahon expectational gaps)
        'governance', 'transparency', 'accountability', 'autonomy',
        'privacy', 'security', 'equity', 'oversight', 'democratic',
        'agency', 'ethical', 'responsible', 'public', 'openness',
        'regulation', 'infrastructure',
        # Expanded: Marginson HE public values + democratic participation
        'safety', 'fairness', 'inclusion', 'access', 'trust',
        'wellbeing', 'diversity', 'participation',
    ]

    values_rows = []
    company_token_counts = {}  # Reuse for absence analysis

    for company in df['Company'].unique():
        if company == "Unknown_Company":
            continue
        company_docs = df[df['Company'] == company]['Processed_Text']
        all_tokens = ' '.join(company_docs).split()
        total_tokens = len(all_tokens)
        token_counts = Counter(all_tokens)
        company_token_counts[company] = (token_counts, total_tokens)

        for term in values_terms:
            count = token_counts.get(term, 0)
            per_1k = round((count / total_tokens) * 1000, 2) if total_tokens > 0 else 0
            values_rows.append({
                'Company': company,
                'Term': term,
                'Count': count,
                'Per1kTokens': per_1k
            })

    values_df = pd.DataFrame(values_rows)
    values_csv_path = os.path.join(OUTPUT_DIR, "values_term_frequency.csv")
    values_df.to_csv(values_csv_path, index=False)
    print(f"\n[OK] Values term frequency saved: {values_csv_path}")
    print(f"    Terms tracked: {len(values_terms)}")
    for company in values_df['Company'].unique():
        top = values_df[values_df['Company'] == company].nlargest(3, 'Per1kTokens')
        top_str = ', '.join(f"{r['Term']}({r['Per1kTokens']}‰)" for _, r in top.iterrows())
        print(f"      {company}: {top_str}")

    # ------------------------------------------
    # AVERAGE TOPIC PROBABILITY PER COMPANY
    # ------------------------------------------
    # Shows company-topic affinity beyond hard assignment counts
    # Captures soft clustering: a doc assigned to Topic 0 may still have
    # 30% probability for Topic 1, revealing latent thematic overlap

    import ast

    companies = [c for c in df['Company'].unique() if c != "Unknown_Company"]
    n_topics = len(topic_model.get_topic_info()[topic_model.get_topic_info().Topic != -1])

    prob_rows = []
    for company in companies:
        company_df = df[df['Company'] == company]
        prob_matrix = []
        for _, row in company_df.iterrows():
            probs_val = row['Topic_Distribution']
            if isinstance(probs_val, str):
                probs_val = ast.literal_eval(probs_val)
            prob_matrix.append(list(probs_val)[:n_topics])

        avg_probs = np.mean(prob_matrix, axis=0)
        for tid in range(n_topics):
            prob_rows.append({
                'Company': company,
                'Topic': tid,
                'Avg_Probability': round(float(avg_probs[tid]), 4),
                'Doc_Count': int((company_df['Topic'] == tid).sum())
            })

    prob_df = pd.DataFrame(prob_rows)
    prob_csv_path = os.path.join(OUTPUT_DIR, "topic_probability_by_company.csv")
    prob_df.to_csv(prob_csv_path, index=False)
    print(f"\n[OK] Topic probability by company saved: {prob_csv_path}")
    for company in companies:
        cp = prob_df[prob_df['Company'] == company]
        probs_str = ', '.join(f"T{r['Topic']}={r['Avg_Probability']:.2%}" for _, r in cp.iterrows())
        print(f"      {company}: {probs_str}")

    # ------------------------------------------
    # ABSENCE ANALYSIS (Expectational Gaps)
    # ------------------------------------------
    # Mahon Type II expectational gap: what companies DON'T say
    # Terms with 0 or near-0 frequency reveal blind spots
    # Threshold: <0.05 per 1k tokens = functionally absent

    ABSENCE_THRESHOLD = 0.05  # per 1k tokens

    absence_rows = []
    for company in companies:
        company_vals = values_df[values_df['Company'] == company]
        absent = company_vals[company_vals['Per1kTokens'] < ABSENCE_THRESHOLD]
        for _, row in absent.iterrows():
            absence_rows.append({
                'Company': company,
                'Absent_Term': row['Term'],
                'Count': row['Count'],
                'Per1kTokens': row['Per1kTokens']
            })

    absence_df = pd.DataFrame(absence_rows)
    absence_csv_path = os.path.join(OUTPUT_DIR, "expectational_gaps.csv")
    absence_df.to_csv(absence_csv_path, index=False)
    print(f"\n[OK] Expectational gaps (absent values terms) saved: {absence_csv_path}")
    for company in companies:
        ca = absence_df[absence_df['Company'] == company]
        if not ca.empty:
            terms = ', '.join(ca['Absent_Term'].tolist())
            print(f"      {company} blind spots: {terms}")

    # ------------------------------------------
    # COMPANY POSITIONING HEATMAP (Interactive HTML)
    # ------------------------------------------
    # Unified comparative visualization: rows=terms, columns=companies
    # Color intensity = normalized frequency (per 1k tokens)
    # Bridges BERTopic topics + values analysis into one thesis-ready figure

    pivot = values_df.pivot(index='Term', columns='Company', values='Per1kTokens').fillna(0)
    pivot = pivot[sorted(pivot.columns)]  # Alphabetical company order

    # Sort terms by max frequency (most discussed first)
    pivot['_max'] = pivot.max(axis=1)
    pivot = pivot.sort_values('_max', ascending=False).drop(columns=['_max'])

    # Generate HTML heatmap
    html_parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        '<title>Company Positioning: Democratic Values Heatmap</title>',
        '<style>',
        'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; ',
        '       background: #1a1a2e; color: #eee; padding: 40px; }',
        'h1 { color: #FFD700; font-size: 1.4em; }',
        'h2 { color: #87CEFA; font-size: 1.1em; font-weight: normal; margin-top: 5px; }',
        'table { border-collapse: collapse; margin: 20px 0; }',
        'th, td { padding: 10px 18px; text-align: center; border: 1px solid #333; }',
        'th { background: #16213e; color: #FFD700; font-weight: 600; }',
        'th.term { text-align: left; min-width: 140px; }',
        'td.term { text-align: left; font-weight: 500; color: #ccc; }',
        'td.absent { color: #ff6b6b; font-style: italic; }',
        '.legend { margin-top: 20px; font-size: 0.85em; color: #888; }',
        '.legend span { display: inline-block; width: 18px; height: 18px; ',
        '               vertical-align: middle; margin-right: 4px; border-radius: 3px; }',
        '.note { margin-top: 30px; padding: 15px; background: #16213e; ',
        '        border-left: 3px solid #FFD700; font-size: 0.9em; }',
        '</style></head><body>',
        '<h1>Company Positioning: Democratic Values Term Frequency</h1>',
        '<h2>Per 1,000 tokens (normalized by company corpus size)</h2>',
        '<table><tr><th class="term">Value Term</th>',
    ]

    for col in pivot.columns:
        html_parts.append(f'<th>{col}</th>')
    html_parts.append('</tr>')

    # Find global max for color scaling
    global_max = pivot.values.max()

    for term, row in pivot.iterrows():
        html_parts.append(f'<tr><td class="term">{term}</td>')
        for company in pivot.columns:
            val = row[company]
            if val < ABSENCE_THRESHOLD:
                html_parts.append(f'<td class="absent" style="background:rgba(255,50,50,0.15);">{val:.2f}</td>')
            else:
                intensity = val / global_max if global_max > 0 else 0
                r_c = int(30 + 50 * (1 - intensity))
                g_c = int(80 + 175 * intensity)
                b_c = int(120 + 80 * intensity)
                html_parts.append(
                    f'<td style="background:rgba({r_c},{g_c},{b_c},0.5); '
                    f'font-weight:{600 if intensity > 0.5 else 400};">{val:.2f}</td>'
                )
        html_parts.append('</tr>')

    html_parts.append('</table>')

    # Legend
    html_parts.append('<div class="legend">')
    html_parts.append('<span style="background:rgba(255,50,50,0.15);border:1px solid #ff6b6b;"></span> ')
    html_parts.append(f'<em>Absent</em> (&lt;{ABSENCE_THRESHOLD} per 1k tokens — expectational gap) &nbsp;&nbsp;')
    html_parts.append('<span style="background:rgba(50,200,180,0.5);"></span> ')
    html_parts.append('<em>Present</em> (color intensity = relative frequency)')
    html_parts.append('</div>')

    # Contextual note
    html_parts.append('<div class="note">')
    html_parts.append('<strong>Reading guide:</strong> Red-highlighted cells indicate terms ')
    html_parts.append('functionally absent from a company\'s discourse — these are ')
    html_parts.append('<em>expectational gaps</em> (Mahon, 2002) where corporate framing ')
    html_parts.append('diverges from democratic public values (Marginson, 2011). ')
    html_parts.append('Higher values indicate stronger discursive emphasis.')
    html_parts.append('</div>')

    html_parts.append('</body></html>')

    heatmap_path = os.path.join(OUTPUT_DIR, "company_values_heatmap.html")
    with open(heatmap_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"\n[OK] Company positioning heatmap saved: {heatmap_path}")

    print("\nPipeline complete.")
