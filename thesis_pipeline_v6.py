import os
import re
import random
import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords

# Topic modeling
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

try:
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    ENGLISH_STOPWORDS = set()

from pipeline_utils import (
    mine_pdfs,
    custom_tokenizer,
    postprocess_topics_csv,
    save_cluster_diagnostics,
    generate_knowledge_graph,
    generate_sociotechnical_dashboard,
    read_finder_comment,
)

# Corpus methodology is specified in the thesis text.
# This script keeps only executable constraints (e.g., configuration hyperparameters)
# to avoid narrative drift between code comments and the written methodology.

# ==========================================
# SETUP
# ==========================================
SEED = 42
UMAP_N_NEIGHBORS = 14
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.35
HDBSCAN_MIN_CLUSTER_SIZE = 4
HDBSCAN_MIN_SAMPLES = 1
VECTORIZER_NGRAM_RANGE = (1, 2)
VECTORIZER_MIN_DF = 1
VECTORIZER_MAX_DF = 1.0  # Disabled; using KeyBERTInspired representation instead of c-TF-IDF
TOPIC_MIN_TOPIC_SIZE = 5
TOPIC_TOP_N_WORDS = 10
TOPIC_TARGET_COUNT = 6
ABSENCE_THRESHOLD = 0.05  # Per 1k tokens

random.seed(SEED)
np.random.seed(SEED)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


# VECTORIZER APPROACH: Minimal Brand Stopwords + NLTK
# - min_df=1: Include all terms (no silent filtering)
# - NLTK stopwords: Standard English noise removal
# - brand_stopwords: Explicit removal of company/product names
# - BERTopic semantic embedding does the heavy lifting for clustering
#
# This gives BERTopic complete access to the corpus semantics while
# explicitly filtering only known noise. All filtering decisions are transparent and justifiable.
brand_stopwords = [
    # ========== Brand/Product names (filter branding noise) ==========
    'anthropic', 'microsoft', 'google', 'facebook', 'amazon',
    'azure', 'openai', 'deepmind', 'claude', 'gpt', 'gemini',
    'copilot', 'linkedin', 'palm', 'learnlm', 'llama',
]

_topic_name_cache = {}


# ==========================================
# MAIN PIPELINE
# ==========================================
# Helper functions for metadata extraction
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


def generate_topic_names(topic_model):
    """
    Generate human-readable names for BERTopic topics using deterministic fallback logic.
    
    Args:
        topic_model: Fitted BERTopic model
    
    Returns:
        dict: Mapping of topic_id to human-readable name
    """
    topic_info = topic_model.get_topic_info()
    
    topic_names = {}
    for idx, row in topic_info.iterrows():
        if row['Topic'] == -1:  # Skip noise
            continue

        topic_id = int(row['Topic'])
        topic_words = topic_model.get_topic(topic_id) or []
        keyword_list = []
        seen_keywords = set()
        for word, _score in topic_words:
            cleaned = re.sub(r"\s+", " ", str(word).strip())
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen_keywords:
                continue
            seen_keywords.add(normalized)
            keyword_list.append(cleaned)

        cache_key = tuple(word.lower() for word in keyword_list)
        if cache_key in _topic_name_cache:
            topic_names[topic_id] = _topic_name_cache[cache_key]
            print(f"[TOPIC] Topic {topic_id}: {topic_names[topic_id]} (cache)")
            continue

        def fallback_topic_name(keywords):
            ordered_terms = []
            seen_terms = set()
            for keyword in keywords:
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", str(keyword)):
                    normalized = token.lower()
                    if len(normalized) < 3:
                        continue
                    if normalized in ENGLISH_STOPWORDS:
                        continue
                    if normalized in seen_terms:
                        continue
                    seen_terms.add(normalized)
                    ordered_terms.append(token.title())
            if not ordered_terms:
                return f"Topic {topic_id}"
            if len(ordered_terms) == 1:
                return ordered_terms[0]
            if len(ordered_terms) == 2:
                return f"{ordered_terms[0]} and {ordered_terms[1]}"
            return " ".join(ordered_terms[:2])

        topic_name = fallback_topic_name(keyword_list)
        topic_names[topic_id] = topic_name
        _topic_name_cache[cache_key] = topic_name
        print(f"[TOPIC] Topic {topic_id}: {topic_name} (fallback)")
    
    return topic_names


if __name__ == "__main__":
    BASE_DIR = "./Thesis_Data_Mining"
    OUTPUT_DIR = os.path.join(BASE_DIR, "04_Analysis_Outputs")
    EXCLUDED_DIR = "./Excluded_Documents_Low_Relevance"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data extraction
    df = mine_pdfs(BASE_DIR)

    if df.empty:
        print("No data found.")
        exit()

    df.to_csv(os.path.join(OUTPUT_DIR, "corpus.csv"), index=False)

    docs = df["Processed_Text"].tolist()

    # ------------------------------------------
    # BERTopic with Hybrid Filtering
    # ------------------------------------------
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=SEED
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # HYBRID APPROACH: Statistical Filtering + Minimal Brand Stopwords
    # - min_df=3: Word must appear in at least 3 documents
    # - max_df=1.0: Disabled to avoid prior c-TF-IDF instability
    # - brand_stopwords: Explicit removal of company/product names that statistically persist
    # 
    # This is more intelligent than pure statistical filtering and requires less manual curation
    # than comprehensive stopword lists.
    
    nltk_stopwords = set(stopwords.words('english'))
    combined_stopwords = list(nltk_stopwords.union(set(brand_stopwords)))
    
    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=VECTORIZER_NGRAM_RANGE,
        min_df=VECTORIZER_MIN_DF,
        max_df=VECTORIZER_MAX_DF,
        stop_words=combined_stopwords  # NLTK + minimal brand names
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        min_topic_size=TOPIC_MIN_TOPIC_SIZE,
        nr_topics=TOPIC_TARGET_COUNT,
        top_n_words=TOPIC_TOP_N_WORDS,
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
    postprocess_topics_csv(topics_csv_path, brand_stopwords)

    # ------------------------------------------
    # Generate topic names (deterministic frequency-based)
    # ------------------------------------------
    print("\n[NAMING] Generating topic names (deterministic approach)...")
    topic_names = generate_topic_names(topic_model)

    # Save named topics
    named_topics_df = topic_info.copy()
    named_topics_df['Human_Name'] = named_topics_df['Topic'].map(topic_names)
    named_topics_csv_path = os.path.join(OUTPUT_DIR, "topics_named.csv")
    named_topics_df.to_csv(named_topics_csv_path, index=False)
    print(f"[OK] Named topics saved: {named_topics_csv_path}")

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
    
    # Extract URLs and publication years from comments
    comment_urls = full_comments.apply(lambda c: parse_url_and_year_from_comment(c)[0])
    comment_years = full_comments.apply(lambda c: parse_url_and_year_from_comment(c)[1])
    
    appendix_df['URL'] = comment_urls
    appendix_df['PublicationYear'] = comment_years
    appendix_df['DateEvidence'] = comment_years.apply(
        lambda y: "finder_comment" if y else "missing"
    )
    appendix_df['DocumentType'] = appendix_df.apply(
        lambda row: classify_document_type(row['FileName'], row['URL']),
        axis=1
    )

    # All documents have been pre-filtered to 2023+ during corpus assembly.
    # Save defensible sampling frame log (with publication metadata for transparency).
    selection_log_df = appendix_df.copy()
    selection_log_df['FilePath'] = selection_log_df.apply(
        lambda row: f"Thesis_Data_Mining/{row['Level']}/{row['Company']}/{row['FileName']}",
        axis=1
    )
    selection_log_df = selection_log_df[[
        'Company', 'Level', 'FileName', 'FilePath', 'DocumentType', 'PublicationYear',
        'DateEvidence', 'URL'
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
                    'ExclusionReason': "Low relevance (manual exclusion folder)",
                    'URL': url_from_comment,
                })

    excluded_low_rel_df = pd.DataFrame(excluded_low_rel_rows)

    # Build a single audit trail across included and excluded documents.
    audit_frames = [selection_log_df.copy()]
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
        '       background: #1a1a2e; color: #eee; padding: 4px 40px 40px; }',
        'h1 { color: #FFD700; font-size: 1.4em; margin: 0; }',
        'h2 { color: #87CEFA; font-size: 1.1em; font-weight: normal; margin: 2px 0 8px; }',
        'table { border-collapse: collapse; margin: 8px 0; }',
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

    # ------------------------------------------
    # Sociotechnical dashboard (combined page)
    # ------------------------------------------
    generate_sociotechnical_dashboard(OUTPUT_DIR)

    print("\nPipeline complete.")
