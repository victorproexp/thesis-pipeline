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

# STATISTICAL FILTERING APPROACH (Option 1)
# Use aggressive min_df and max_df parameters to let statistics decide what's noise.
# Words appearing in too few/many documents are automatically filtered.
# This is smarter than curated stopwords — no manual list needed.

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

    # ------------------------------------------
    # Company→Topic Edges (Word Count Weighting + Soft Clustering)
    # ------------------------------------------
    # Assign topics and probabilities
    df["Topic"] = topic_model.topics_
    df["Topic_Distribution"] = list(topic_model.probabilities_)
    
    threshold_prob = 0.25  # Minimum 25% probability to show edge

    for idx, row in df.iterrows():
        company = str(row["Company"])
        word_count = int(row["Word_Count"]) if "Word_Count" in row else len(str(row["Processed_Text"]).split())
        
        # Soft clustering: iterate through all topics with non-zero probability
        probs = row["Topic_Distribution"]
        
        for topic_id, prob in enumerate(probs):
            if prob > threshold_prob:  # Only show significant connections
                # Edge weight = (probability * word_count) / 100
                # This ensures longer documents pull harder on topics they're aligned with
                edge_weight = (prob * word_count) / 100
                
                net.add_edge(
                    company,
                    int(topic_id),
                    value=float(edge_weight),
                    title=f"Strength: {float(edge_weight):.1f} (prob: {prob:.0%})"
                )

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
    # BERTopic with Statistical Filtering
    # ------------------------------------------
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    umap_model = UMAP(
        n_neighbors=18,        # TUNED: Increased from 8 → focus on global structure vs local noise
        n_components=5,
        min_dist=0.35,         # TUNED: Increased from 0.2 → spread points apart for better cluster separation
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=5,    # Lowered from 12 to allow smaller focused clusters (more topics)
        min_samples=3,         # Keep stricter density requirement
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # STATISTICAL FILTERING VIA AGGRESSIVE MIN_DF / MAX_DF
    # Let CountVectorizer's statistical thresholds handle noise filtering.
    # - min_df=2: Word must appear in at least 2 documents (filters out rare/document-specific terms)
    # - max_df=0.80: Word appears in at most 80% of documents (filters out overly common corpus-wide noise)
    # This approach requires NO manual stopwords list — statistics decide what's relevant.
    
    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=(1, 2),
        min_df=2,             # Word must appear in ≥2 documents (filters rare terms)
        max_df=0.80,          # Word can appear in ≤80% of documents (aggressive — filters common noise)
        stop_words='english'  # NLTK English stopwords only (no custom list)
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        min_topic_size=3,      # Lowered to extract more granular SI narratives (target 5-8 topics)
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
