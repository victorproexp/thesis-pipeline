import os
import re
import subprocess
import plistlib
import unicodedata
import ast

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

from pyvis.network import Network


lemmatizer = WordNetLemmatizer()


def remove_urls(text: str) -> str:
    """Remove URLs and web references"""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
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
    text = remove_urls(text)
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


def mine_pdfs(base_dir):
    """
    Extract PDFs WITHOUT chunking - preserve full document context.
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


def custom_tokenizer(text):
    return [
        word.lower() for word in text.split()
        if len(word) > 2
    ]


def clean_topic_words(words_str: str, max_words: int = 10) -> str:
    """Post-process topic words for CSV output."""
    try:
        words = ast.literal_eval(words_str)
    except Exception:
        words = [w.strip().strip("'") for w in words_str.strip("[]").split(",")]

    cleaned = []
    seen_prefixes = set()

    for phrase in words:
        if not isinstance(phrase, str):
            continue

        phrase_words = phrase.split()

        for word in phrase_words:
            if len(word) <= 2 or not word.isalpha():
                continue

            word = word.lower().strip()

            if word in {'education', 'learning', 'teaching', 'student', 'educator', 'initiative'}:
                if word in seen_prefixes:
                    continue
                seen_prefixes.add(word)

            if word not in cleaned:
                cleaned.append(word)

    return str(cleaned[:max_words])


def postprocess_topics_csv(csv_path: str, brand_stopwords):
    """Clean representation terms in topics CSV."""
    df = pd.read_csv(csv_path, dtype={'Representation': str})

    def remove_stopwords_from_rep(rep_str: str, stopwords_set: set) -> str:
        try:
            words = ast.literal_eval(rep_str)
            if not isinstance(words, list):
                return rep_str

            filtered = [w for w in words if isinstance(w, str) and w.lower() not in stopwords_set]
            seen = set()
            unique_filtered = []
            for w in filtered:
                w_lower = w.lower()
                if w_lower not in seen:
                    unique_filtered.append(w)
                    seen.add(w_lower)

            return str(unique_filtered[:10])
        except Exception:
            return rep_str

    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = nltk_stopwords.union(set(brand_stopwords))

    df['Representation'] = df['Representation'].apply(
        lambda x: remove_stopwords_from_rep(x, all_stopwords)
    )

    df['Representation'] = df['Representation'].apply(
        lambda x: clean_topic_words(x, max_words=10)
    )

    df.to_csv(csv_path, index=False)
    print(f"[OK] Topics cleaned (stopwords removed) and saved to {csv_path}")


def save_cluster_diagnostics(corpus_with_topics_path: str, topics_csv_path: str, output_path: str):
    """Compute and save BERTopic cluster diagnostics."""
    corpus_df = pd.read_csv(corpus_with_topics_path)
    topics_df = pd.read_csv(topics_csv_path)

    doc_count = len(corpus_df)
    noise_docs = int((corpus_df["Topic"] == -1).sum())
    noise_ratio = float(noise_docs / doc_count) if doc_count else 0.0

    non_noise_df = corpus_df[corpus_df["Topic"] != -1].copy()
    num_topics_excl_noise = int(topics_df[topics_df["Topic"] != -1]["Topic"].nunique())

    counts = non_noise_df["Topic"].value_counts().sort_index()
    if len(counts) > 0 and counts.sum() > 0:
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        max_entropy = float(np.log(len(p))) if len(p) > 1 else 1.0
        topic_balance_entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        topic_balance_entropy_norm = 0.0

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

    def _parse_probs(raw_value):
        arr = np.fromstring(str(raw_value).strip("[]"), sep=" ")
        return arr if arr.size else np.array([])

    max_probs = []
    for raw_probs in corpus_df["Topic_Distribution"].dropna().tolist():
        arr = _parse_probs(raw_probs)
        if arr.size:
            max_probs.append(float(arr.max()))
    mean_max_topic_probability = float(np.mean(max_probs)) if max_probs else np.nan

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


def generate_knowledge_graph(df, topic_model, output_path):
    """Generate knowledge graph with aggregated company-topic edges."""
    VALUES_TERMS = {
        'governance', 'transparency', 'accountability', 'autonomy',
        'privacy', 'security', 'equity', 'oversight', 'democratic',
        'agency', 'ethical', 'responsible', 'public', 'openness',
        'regulation', 'infrastructure', 'safety', 'fairness',
        'inclusion', 'access', 'trust', 'wellbeing', 'diversity',
        'participation',
    }
    COMPANY_COLORS = {
        "Google": {"node": "#4285F4", "edge": "#4285F4"},
        "Microsoft": {"node": "#00A4EF", "edge": "#00A4EF"},
        "Anthropic": {"node": "#D97757", "edge": "#D97757"},
    }
    TOPIC_COLOR = "#C4A7E7"
    BG_COLOR = "#1a1a2e"

    net = Network(height="800px", width="100%", bgcolor=BG_COLOR, font_color="white")

    company_totals = {}
    for company in df["Company"].unique():
        if company == "Unknown_Company":
            continue
        company_df = df[df["Company"] == company]
        total_words = company_df["Word_Count"].sum() if "Word_Count" in company_df.columns else sum(
            len(str(t).split()) for t in company_df["Raw_Text"]
        )
        company_totals[company] = total_words

    if company_totals:
        max_vol = max(company_totals.values())
        min_vol = min(company_totals.values())
        vol_range = max_vol - min_vol if max_vol != min_vol else 1
    else:
        max_vol = min_vol = 0
        vol_range = 1

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

    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info.Topic != -1]

    for _, trow in valid.iterrows():
        topic_id = trow["Topic"]
        doc_count = trow["Count"]
        words = topic_model.get_topic(topic_id)[:8]
        seen_parts = set()
        deduped = []
        for w, s in words:
            if ' ' in w:
                seen_parts.update(w.split())
                deduped.append((w, s))
            elif w not in seen_parts:
                deduped.append((w, s))
            if len(deduped) == 5:
                break

        all_words = topic_model.get_topic(topic_id)[:15]
        values_in_topic = []
        for w, s in all_words:
            parts = w.split() if ' ' in w else [w]
            for p in parts:
                if p in VALUES_TERMS and p not in values_in_topic:
                    values_in_topic.append(p)

        keyword_line = " · ".join([w for w, _ in deduped])
        values_line = f"⟨{', '.join(values_in_topic)}⟩" if values_in_topic else ""
        label = f"T{topic_id}\n{keyword_line}"
        if values_line:
            label += f"\n{values_line}"

        size = 15 + doc_count * 1.2
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

    total_corpus_words = sum(company_totals.values()) if company_totals else 1

    edge_data = {}
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

    threshold_prob = 0.15

    for (company, topic_id), stats in edge_data.items():
        avg_prob = stats["prob_sum"] / stats["doc_count"]
        if avg_prob < threshold_prob:
            continue
        discourse_share = stats["word_sum"] / total_corpus_words
        edge_weight = avg_prob * discourse_share * 300

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

    net.write_html(output_path)

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
    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    freeze_js = """
    <script type="text/javascript">
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
    # Remove default pyvis card wrapper/border so the graph blends with the dashboard background.
    html = html.replace('<center>\n<h1></h1>\n</center>\n\n', '')
    html = html.replace('\n\n        <center>\n          <h1></h1>\n        </center>', '')
    html = html.replace('<body>', '<body style="margin:0;background-color:#1a1a2e;">')
    html = html.replace("border: 1px solid lightgray;", "border: 0;")
    html = html.replace('<div class="card" style="width: 100%">', '')
    html = html.replace('class="card-body"', '')
    html = html.replace('</div>\n\n        \n        \n\n        <script type="text/javascript">',
                        '\n\n        \n        \n\n        <script type="text/javascript">')
    html = html.replace('<div id="mynetwork"', legend_html + '\n        <div id="mynetwork"')
    html = html.replace('</body>', freeze_js + '\n</body>')
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_sociotechnical_dashboard(output_dir):
    """Generate combined thesis dashboard page embedding heatmap + knowledge graph."""
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>The Sociotechnical Imaginary of AI in Higher Education</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --bg-elev: #162338;
            --bg-soft: #1f2f4a;
            --text: #e7edf8;
            --muted: #aebcd4;
            --accent: #79d7c2;
            --accent-2: #f0c36d;
            --border: rgba(255, 255, 255, 0.14);
            --shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            color: var(--text);
            background: var(--bg);
            font-family: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.55;
        }

        .container {
            width: min(1440px, 94vw);
            margin: 0 auto;
            padding: 32px 0 48px;
        }

        .hero {
            background: linear-gradient(145deg, rgba(31, 47, 74, 0.9), rgba(22, 35, 56, 0.94));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 26px 26px 22px;
            box-shadow: var(--shadow);
        }

        .kicker {
            font-size: 0.84rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent);
            font-weight: 700;
            margin-bottom: 8px;
        }

        h1 {
            margin: 0;
            font-size: clamp(1.5rem, 1.9vw + 1rem, 2.35rem);
            line-height: 1.15;
        }

        .hero p {
            margin: 12px 0 0;
            color: var(--muted);
            max-width: 95ch;
            font-size: 0.99rem;
        }

        .lens-grid {
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .lens-card {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            padding: 12px 14px;
        }

        .lens-card h2 {
            margin: 0 0 6px;
            color: var(--accent-2);
            font-size: 1rem;
        }

        .lens-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .viz-grid {
            margin-top: 12px;
            display: grid;
            grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
            gap: 18px;
            align-items: start;
            width: 100%;
            margin-left: 0;
        }

        .right-stack {
            display: flex;
            flex-direction: column;
            gap: 0;
        }

        .viz-panel {
            background: transparent;
            border: 0;
            border-radius: 0;
            overflow: visible;
            box-shadow: none;
        }

        .viz-panel header {
            padding: 2px 14px 4px;
            border-bottom: 0;
            background: transparent;
        }

        .viz-panel h3 {
            margin: 0;
            font-size: 1rem;
            line-height: 1.25;
        }

        .viz-panel p {
            margin: 2px 0 0;
            color: var(--muted);
            font-size: 0.9rem;
        }

        iframe {
            display: block;
            width: 100%;
            height: 920px;
            border: 0;
            background: var(--bg);
            margin: 0;
            overflow: hidden;
        }

        .heatmap-frame {
            height: 1880px;
        }

        .graph-frame {
            height: 920px;
        }

        .synthesis {
            margin-top: 0;
            background: var(--bg-elev);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }

        .synthesis h4 {
            margin: 0 0 8px;
            color: var(--accent-2);
            font-size: 1rem;
        }

        .synthesis ul {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
            font-size: 0.93rem;
        }

        .meta {
            margin-top: 12px;
            color: var(--muted);
            font-size: 0.84rem;
        }

        @media (max-width: 1120px) {
            .lens-grid {
                grid-template-columns: 1fr;
            }

            .viz-grid {
                grid-template-columns: 1fr;
                gap: 14px;
            }

            iframe {
                height: 900px;
            }

            .heatmap-frame {
                height: 1780px;
            }

            .graph-frame {
                height: 900px;
            }
        }

        @media (max-width: 700px) {
            .container {
                width: 95vw;
                padding: 20px 0 34px;
            }

            .hero {
                padding: 20px 16px;
            }

            iframe {
                height: 840px;
            }

            .heatmap-frame {
                height: 1680px;
            }

            .graph-frame {
                height: 840px;
            }
        }
    </style>
</head>
<body>
    <main class="container">
        <section class="hero">
            <div class="kicker">Analytical Figure Dashboard</div>
            <h1>The Sociotechnical Imaginary of AI in Higher Education</h1>
            <p>
                This page integrates lexical value salience and topic-network structure across Anthropic, Google, and Microsoft documents in the 2023 to 2026 corpus.
                The analytical framing follows the thesis lens: sociotechnical imaginaries as publicly performed visions, expectational gaps between what is and what ought to be,
                and the connector-complementor tension in higher-education governance.
            </p>

            <div class="lens-grid">
                <article class="lens-card">
                    <h2>Sociotechnical Imaginary</h2>
                    <p>
                        Recurring value language and topic emphasis are interpreted as institutionally stabilized visions of desirable educational futures.
                    </p>
                </article>
                <article class="lens-card">
                    <h2>Expectational Gap</h2>
                    <p>
                        Weak representation of democratic public values is contrasted with strong infrastructural and performance-oriented framing.
                    </p>
                </article>
                <article class="lens-card">
                    <h2>Issue Management Dynamics</h2>
                    <p>
                        Discursive positioning is assessed in relation to potential, imminent, current, and critical issue stages around legitimacy and regulation.
                    </p>
                </article>
            </div>
        </section>

        <section class="viz-grid" aria-label="Combined visualization area">
            <article class="viz-panel">
                <header>
                    <h3>Democratic Public Values Lexical Heatmap</h3>
                    <p>Normalized term frequency per 1,000 tokens across corporate narratives (company-level corpus normalization).</p>
                </header>
                <iframe id="heatmapFrame" class="heatmap-frame" src="company_values_heatmap.html" title="Company values heatmap" scrolling="no"></iframe>
            </article>

            <div class="right-stack">
                <article class="viz-panel">
                    <header>
                        <h3>Company Topic Knowledge Graph</h3>
                        <p>Company-topic coupling by discourse share and average topic probability.</p>
                    </header>
                    <iframe id="graphFrame" class="graph-frame" src="knowledge_graph.html" title="Company topic knowledge graph" scrolling="no"></iframe>
                </article>

                <section class="synthesis">
                    <h4>Interpretive Guide and Analytical Limits</h4>
                    <ul>
                        <li>The heatmap indicates which democratic public values are foregrounded, backgrounded, or effectively absent by company.</li>
                        <li>The graph locates where these value patterns are anchored in broader topical clusters such as security, governance, productivity, and policy.</li>
                        <li>Strong infrastructural framing combined with weak public-value emphasis is treated as a candidate value-conformance gap in Mahon's sense.</li>
                        <li>Lexical frequency is interpreted as discursive salience, not as direct evidence of organizational commitment or implementation outcomes.</li>
                        <li>Findings should be triangulated with qualitative close reading, given differences in document genre and publication format across firms.</li>
                    </ul>
                    <div class="meta">
                        Context alignment: Jasanoff and Kim (2015), Marginson (2011), van Dijck et al. (2018), Coombs (2021), Mahon (2022), Mager and Katzenbach (2021).
                    </div>
                </section>
            </div>
        </section>
    </main>

</body>
</html>
"""
    dashboard_path = os.path.join(output_dir, "sociotechnical_imaginary_dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"[OK] Sociotechnical dashboard saved: {dashboard_path}")
