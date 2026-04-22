import os
import re
import random
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
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
from pyvis.network import Network

# ==========================================
# SETUP
# ==========================================
random.seed(42)
np.random.seed(42)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

# ==========================================
# STOPWORDS STRATEGY
# ==========================================
# This pipeline uses a balanced stopwords approach:
#
# 1. BASE STOPWORDS (198 words)
#    - NLTK English stopwords: articles, prepositions, pronouns, auxiliaries
#    - Examples: "a", "the", "is", "are", "and", "or", "but"
#    - Rationale: These are grammatically necessary but topically empty
#
# 2. CUSTOM STOPWORDS (50 words) - Evidence-based removals:
#    - Generic enterprise noise: "use", "provide", "system", "platform", etc.
#      → Rationale: Too vague to distinguish topics in AI/education context
#    - Web artifacts: "http", "www", "com", "org", "url", etc.
#      → Rationale: Contamination from web-scraped PDF metadata
#    - Noisy fragments: "side", "possible", "notebooklm", "thing", etc.
#      → Rationale: Actual text fragments observed in corpus that don't
#                   contribute to topic coherence
#    - Company names: "anthropic", "microsoft", "google", etc.
#      → Rationale: Prevent company affiliation from drowning out topics
#
# DESIGN NOTES:
# - Words like "make", "enable", "innovation", "leadership" REMOVED
#   because they appeared frequently but didn't degrade topic quality
# - Words like "ai", "artificial", "intelligence" REMOVED
#   because they're so common in the corpus they don't discriminate
# - Words like "day", "time", "year", "data", "information" REMOVED
#   because they're used in almost every document
# - Domain-specific words (education, teaching, learning, student, etc.)
#   kept in the corpus to preserve topical coherence
#
# Total effective stopwords: 198 (NLTK) + 50 (custom) = 248 words

base_stopwords = set(stopwords.words("english"))

# Custom stopwords organized by category with rationale
custom_stopwords = {
    # === GENERIC ENTERPRISE NOISE ===
    # Common vague words that don't distinguish topics in AI/education context
    "use", "provide", "help", "support",
    "system", "systems", "platform", "platforms",
    "solution", "solutions", "business",
    
    # === OVERLY BROAD/META TERMS ===
    # Too general or meta to be informative about actual topics
    "organization", "organizations", "process", "processes",
    "work", "team", "teams", "people", "education",
    
    # === WEB ARTIFACTS ===
    # Remove technical/structural web elements that contaminate text from web scraping
    "http", "https", "www", "com", "org", "edu", "net",
    "html", "url", "site", "page", "blog",
    "bit", "ly",
    
    # === NOISY FRAGMENTS & INCOMPLETE PHRASES ===
    # Text fragments, single-syllable noise, and words that appear in context
    # but don't contribute semantic meaning
    "side", "possible", "notebooklm",  # Actually observed noise words
    "thing", "part", "area",           # Incomplete references
    
    # === COMPANY-SPECIFIC ENTITIES (Context-aware) ===
    # Remove company names and products when they dominate non-topical discussions
    "anthropic", "microsoft", "google", "facebook", "amazon",
    "azure", "openai",  # Microsoft product and AI firm
}

stop_words = base_stopwords.union(custom_stopwords)

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


# ==========================================
# PDF MINING
# ==========================================
def mine_pdfs(base_dir):
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
                    "Processed_Text": processed
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
        word for word in text.split()
        if word not in stop_words and len(word) > 2
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
    Post-process saved topics.csv to clean word representations
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path, dtype={'Representation': str})
    
    # Apply cleaning to Representation column
    df['Representation'] = df['Representation'].apply(
        lambda x: clean_topic_words(x, max_words=10)
    )
    
    # Save cleaned version
    df.to_csv(csv_path, index=False)
    print(f"[OK] Topics cleaned and saved to {csv_path}")


# ==========================================
# KNOWLEDGE GRAPH
# ==========================================
def generate_knowledge_graph(df, topic_model, output_path):
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    # Companies
    for c in df["Company"].unique():
        if c != "Unknown_Company":
            net.add_node(c, label=c, color="#FFD700", size=40)

    # Topics
    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info.Topic != -1]

    for topic_id in valid.Topic:
        words = topic_model.get_topic(topic_id)[:3]
        label = " | ".join([w for w, _ in words])

        net.add_node(int(topic_id), label=label, color="#87CEFA", size=25)

    # Company → Topic edges
    df["Topic"] = topic_model.topics_
    grouped = (
        df[df.Topic != -1]
        .groupby(["Company", "Topic"])
        .size()
        .reset_index(name="Weight")
    )

    for _, row in grouped.iterrows():
        net.add_edge(row["Company"], int(row["Topic"]), value=int(row["Weight"]))

    net.repulsion(node_distance=250)
    net.write_html(output_path)


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    BASE_DIR = "./Thesis_Data_Mining"
    OUTPUT_DIR = os.path.join(BASE_DIR, "04_Analysis_Outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Data extraction
    df = mine_pdfs(BASE_DIR)

    if df.empty:
        print("No data found.")
        exit()

    df.to_csv(os.path.join(OUTPUT_DIR, "corpus.csv"), index=False)

    docs = df["Processed_Text"].tolist()

    # ------------------------------------------
    # Models
    # ------------------------------------------
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    umap_model = UMAP(
        n_neighbors=8,
        n_components=5,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=(1, 2),  # Reduced from (1,3) to avoid complex multi-word phrases
        min_df=2,             # Increased from 1 to filter rare terms
        max_df=0.95           # Remove very common terms
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        min_topic_size=7,      # Increased from 6 to merge very small topics
        top_n_words=10,        # Explicitly set top words to extract
        verbose=True
    )

    # ------------------------------------------
    # Train model
    # ------------------------------------------
    topics, probs = topic_model.fit_transform(docs)

    df["Topic_Distribution"] = list(probs)

    # Save topics
    topic_info = topic_model.get_topic_info()
    topics_csv_path = os.path.join(OUTPUT_DIR, "topics.csv")
    topic_info.to_csv(topics_csv_path, index=False)

    # ------------------------------------------
    # Post-process topics (clean words in CSV)
    # ------------------------------------------
    postprocess_topics_csv(topics_csv_path)

    print("\nTop topics:")
    print(topic_info.head(10))

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

    print("\nPipeline complete.")