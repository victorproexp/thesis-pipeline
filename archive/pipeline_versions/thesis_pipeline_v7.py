import os
import re
import random
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import sys

# ==========================================
# THESIS PIPELINE V7: LLM-BASED REPRESENTATION
# ==========================================
# Description:
#   Alternative to v6 using distilgpt2 (LLM) for topic representation
#   instead of KeyBERTInspired (embedding-based keyword extraction).
#
# Environment:
#   - Optimized for Google Colab (GPU-accelerated, auto-mounted Drive)
#   - Falls back to local execution if not in Colab
#
# To run in Colab:
#   1. Create a new notebook
#   2. Cell 1: !pip install bertopic sentence-transformers umap-learn hdbscan PyMuPDF nltk pyvis -q
#   3. Cell 2: Upload this script or mount Drive and execute
#   4. Set GPU: Runtime → Change runtime type → GPU (T4)
#
# Note: First run downloads distilgpt2 (~353MB) and all-mpnet-base-v2 (~450MB)
# ==========================================

# ==========================================
# COLAB SETUP
# ==========================================
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/Speciale/Thesis_Data_Mining"
    OUTPUT_DIR = os.path.join(BASE_DIR, "04_Analysis_Outputs")
else:
    BASE_DIR = "./Thesis_Data_Mining"
    OUTPUT_DIR = os.path.join(BASE_DIR, "04_Analysis_Outputs")

# ==========================================
# INSTALLATION NOTES FOR COLAB
# ==========================================
# Run this in Colab first cell:
# !pip install bertopic sentence-transformers umap-learn hdbscan PyMuPDF nltk pyvis -q
# Then run this script in the next cell
# ==========================================

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Topic modeling
from bertopic import BERTopic
from bertopic.representation import TextGeneration
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from transformers import pipeline as hf_pipeline

# Visualization
from pyvis.network import Network

# ==========================================
# CORPUS CONSTRUCTION: INCLUSION CRITERIA
# ==========================================
# As per methodology section: systematic corpus with explicit inclusion criteria
# to ensure validity and address sampling bias concerns.
#
# INCLUSION CRITERIA:
#   - Time Window: January 2023 – March 2026 (Post-ChatGPT era)
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
    "time_window": "2023-01-01 to 2026-03-31",
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

# Use only NLTK English stopwords (no custom brand/domain filtering)
nltk_stopwords = set(stopwords.words('english'))

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
    Post-process saved topics.csv to:
    1. Remove any residual stopwords from LLM representation
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
    
    # Use only NLTK stopwords for post-processing
    all_stopwords = set(stopwords.words('english'))
    
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


# ==========================================
# KNOWLEDGE GRAPH
# ==========================================
def generate_knowledge_graph(df, topic_model, output_path):
    """
    Generate knowledge graph with NORMALIZED PROPORTIONAL WEIGHTING.
    
    Addresses supervisor concern: Rather than forcing equal numbers of documents,
    edges are weighted proportionally to reflect each company's discourse share.
    
    Methodology:
      1. Calculate total word count per company (total discourse volume)
      2. For each document, normalize its contribution relative to its company's total
      3. Edge weight = probability × normalized_base_weight × scaling factor
      
    Result: Edge thickness visually represents proportional discourse contribution.
            If Google publishes 10x more relevant documents, that is visible in the graph.
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    # Companies
    for c in df["Company"].unique():
        if c != "Unknown_Company":
            net.add_node(c, label=c, color="#FFD700", size=40)

    # Topics - show top 4 words for better differentiation
    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info.Topic != -1]

    for topic_id in valid.Topic:
        words = topic_model.get_topic(topic_id)[:4]  # Increased from 3 to 4 words
        label = " | ".join([w for w, _ in words])
        net.add_node(int(topic_id), label=label, color="#87CEFA", size=25)

    # ------------------------------------------
    # Company→Topic Edges (Min-Max Normalization)
    # ------------------------------------------
    # Step 1: Calculate total discourse volume per company & document size ranges
    # This addresses the proportionality concern: we measure each company's actual output
    company_totals = {}
    company_doc_ranges = {}
    
    for company in df['Company'].unique():
        if company != "Unknown_Company":
            company_df = df[df['Company'] == company]
            # Total word count = sum of word counts across all documents from this company
            total_vol = company_df['Word_Count'].sum() if 'Word_Count' in company_df.columns else sum(
                [len(str(text).split()) for text in company_df['Raw_Text']]
            )
            company_totals[company] = total_vol
            
            # Get min/max document sizes for min-max scaling
            doc_sizes = company_df['Word_Count'].values if 'Word_Count' in company_df.columns else [
                len(str(text).split()) for text in company_df['Raw_Text']
            ]
            company_doc_ranges[company] = {
                'min': min(doc_sizes),
                'max': max(doc_sizes)
            }
    
    # Assign topics and probabilities
    df["Topic"] = topic_model.topics_
    df["Topic_Distribution"] = list(topic_model.probabilities_)
    
    threshold_prob = 0.25  # Minimum 25% probability to show edge

    # Step 2: Add edges with min-max normalization for better within-company differentiation
    # Min-max scaling spreads document weights across 0.2–0.9 range within each company
    # This gives much better visual differentiation than linear (which compresses to 0.8–0.9)
    
    for idx, row in df.iterrows():
        company = str(row["Company"])
        if company == "Unknown_Company":
            continue
            
        word_count = int(row["Word_Count"]) if "Word_Count" in row else len(str(row["Processed_Text"]).split())
        
        # Min-max scaling: normalize to [0.2, 0.9] range within company
        # If only 1 doc in company, give it full weight of 0.9
        doc_range = company_doc_ranges.get(company, {'min': word_count, 'max': word_count})
        if doc_range['max'] == doc_range['min']:
            normalized_base = 0.9
        else:
            normalized_base = 0.2 + 0.7 * (word_count - doc_range['min']) / (doc_range['max'] - doc_range['min'])
        
        # Soft clustering: iterate through all topics with non-zero probability
        probs = row["Topic_Distribution"]
        
        for topic_id, prob in enumerate(probs):
            if prob > threshold_prob:  # Only show significant connections
                # Edge weight = probability × normalized_base × scaling factor
                # This ensures:
                #   - Stronger topics (higher prob) have thicker edges
                #   - Within each company, documents are differentiated across 0.2–0.9
                #   - Natural imbalances are preserved as meaningful data
                edge_weight = prob * normalized_base * 100
                
                net.add_edge(
                    company,
                    int(topic_id),
                    value=float(edge_weight),
                    title=f"{company} → Topic {topic_id}\n" +
                          f"Topic strength: {prob:.0%}\n" +
                          f"Document weight: {normalized_base:.1%}\n" +
                          f"Edge thickness: {edge_weight:.1f}"
                )

    net.repulsion(node_distance=250)
    net.write_html(output_path)


# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    # BASE_DIR and OUTPUT_DIR already set above for Colab/local
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if IS_COLAB:
        print("🔍 Running on Google Colab (GPU-enabled)")
    else:
        print("🐍 Running locally")

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

    # STATISTICAL FILTERING ONLY
    # - min_df=2: Word must appear in at least 2 documents
    # - max_df=0.90: Word appears in at most 90% of documents
    # - No custom brand/domain stopwords - let data speak naturally
    
    vectorizer_model = CountVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=(1, 2),
        min_df=2,             # Word must appear in ≥2 documents
        max_df=0.90,          # Word can appear in ≤90% of documents
        stop_words=list(nltk_stopwords)  # NLTK English stopwords only
    )

    # ------------------------------------------
    # LLM-based Representation (Interpretation 2)
    # Uses distilgpt2 (smallest causal LM) to generate topic descriptions
    # instead of keyword-based KeyBERTInspired
    # Note: flan-t5 requires seq2seq pipeline (not supported by BERTopic)
    #       so we use distilgpt2 which works with text-generation pipeline
    # ------------------------------------------
    prompt = "Topic keywords: [KEYWORDS]"

    # Auto-detect GPU availability
    device = 0 if IS_COLAB else -1  # GPU on Colab, CPU on local

    generator = hf_pipeline(
        "text-generation",
        model="distilgpt2",
        device=device
    )

    representation_model = TextGeneration(
        generator,
        prompt=prompt,
        doc_length=100,  # Increased for GPU (Colab can handle longer context)
        tokenizer="whitespace"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        min_topic_size=4,
        top_n_words=10,
        verbose=True
    )

    # ------------------------------------------
    # Train model
    # ------------------------------------------
    topics, probs = topic_model.fit_transform(docs)

    df["Topic_Distribution"] = list(probs)

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

    # ------------------------------------------
    # Generate Appendix: Corpus Metadata
    # ------------------------------------------
    # As per supervisor feedback: transparency in sampling process via
    # comprehensive metadata table listing all documents.
    # This table should appear in thesis appendix.
    
    appendix_df = df[['Company', 'Level', 'FileName', 'Raw_Text']].copy()
    appendix_df['WordCount'] = df['Raw_Text'].apply(lambda x: len(str(x).split()))
    
    # Reorder columns for readability
    appendix_df = appendix_df[['Company', 'Level', 'FileName', 'WordCount', 'Raw_Text']]
    
    # Add a document URL/path column for reference (currently filename only)
    appendix_df['FilePath'] = df.apply(
        lambda row: f"Thesis_Data_Mining/{row['Level']}/{row['Company']}/{row['FileName']}",
        axis=1
    )
    
    appendix_csv_path = os.path.join(OUTPUT_DIR, "appendix_corpus_metadata.csv")
    appendix_df.to_csv(appendix_csv_path, index=False)
    print(f"[OK] Appendix metadata saved: {appendix_csv_path}")
    print(f"    Total documents: {len(appendix_df)}")
    print(f"    Documents per company:")
    for company in appendix_df['Company'].unique():
        if company != "Unknown_Company":
            count = len(appendix_df[appendix_df['Company'] == company])
            total_words = appendix_df[appendix_df['Company'] == company]['WordCount'].sum()
            print(f"      - {company}: {count} documents, {total_words:,} words")

    print("\nPipeline complete.")
