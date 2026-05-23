import os
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
    generate_topic_names,
    build_appendix_metadata,
    compute_values_term_frequency,
    compute_topic_probability_by_company,
    generate_company_values_heatmap,
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


# ==========================================
# MAIN PIPELINE
# ==========================================


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
    appendix_df, selection_log_df, excluded_low_rel_df, corpus_audit_df = build_appendix_metadata(
        df,
        BASE_DIR,
        EXCLUDED_DIR,
    )
    
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
    values_df, company_token_counts = compute_values_term_frequency(df, values_terms)
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
    prob_df = compute_topic_probability_by_company(df, topic_model)
    prob_csv_path = os.path.join(OUTPUT_DIR, "topic_probability_by_company.csv")
    prob_df.to_csv(prob_csv_path, index=False)
    print(f"\n[OK] Topic probability by company saved: {prob_csv_path}")
    companies = [c for c in df['Company'].unique() if c != "Unknown_Company"]
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
    heatmap_path = os.path.join(OUTPUT_DIR, "company_values_heatmap.html")
    generate_company_values_heatmap(values_df, heatmap_path, ABSENCE_THRESHOLD)
    print(f"\n[OK] Company positioning heatmap saved: {heatmap_path}")

    # ------------------------------------------
    # Sociotechnical dashboard (combined page)
    # ------------------------------------------
    generate_sociotechnical_dashboard(OUTPUT_DIR)

    print("\nPipeline complete.")
