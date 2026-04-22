import os
import re
import pandas as pd
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords, words
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from pyvis.network import Network
import numpy as np
import random
from umap import UMAP

import nltk
nltk.download('wordnet', quiet=True)

# ==========================================
# SETUP: NLTK AND ADVANCED ENTITY MASKING
# ==========================================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# THE SCIENTIFIC MASKING DICTIONARY
# Stripping brands, technology jargon, and education-sector noise to expose corporate values
custom_stops = {
    # ==================================================
    # BRAND & COMPANY NAMES
    # ==================================================
    'google','microsoft','openai','anthropic',
    'chatgpt','copilot','gemini','claude',
    
    # ==================================================
    # CORE AI/TECHNOLOGY TERMINOLOGY
    # ==================================================
    'ai', 'ml', 'digital', 'technology', 'technologies', 'tech', 'platform', 'platforms',
    'system', 'systems', 'tool', 'tools', 'framework', 'frameworks', 'software',
    
    # ==================================================
    # DEVELOPMENT & IMPLEMENTATION VERBS/NOUNS
    # ==================================================
    'develop', 'develops', 'development', 'develop', 'develop',
    'implement', 'implements', 'implementing',
    'build', 'builds', 'create', 'creates',
    'application', 'applications', 'solution', 'solutions',
    'approach', 'approaches', 'process', 'processes',
    
    # ==================================================
    # EDUCATION SECTOR SPECIFIC TERMS
    # ==================================================
    'education', 'educational',
    'learn', 'learning', 'learns', 'teaching', 'teachings', 'teach',
    'student', 'students', 'teacher', 'teachers',
    'educator', 'educators', 'faculty', 'faculties',
    'school', 'schools', 'college', 'colleges', 'university', 'universities',
    'academic', 'campuses', 'training',
    'quiz', 'questions', 'question', 'study',
    
    # ==================================================
    # RESEARCH & DATA TERMINOLOGY
    # ==================================================
    'research', 'researches', 'data', 'analysis', 'analytics',
    'insight', 'insights', 'information', 'informations',
    
    # ==================================================
    # ROLES & POSITIONS
    # ==================================================
    'developer', 'developers', 'users', 'user',
    'assistant', 'assistants', 'agent', 'agents',
    'employee', 'employees', 'scientist', 'scientists',
    'leader', 'creators',
    
    # ==================================================
    # GENERIC ACTION VERBS & FILLER WORDS
    # ==================================================
    'work', 'works', 'working', 'need', 'needs', 'help', 'helps',
    'support', 'supports', 'use', 'uses', 'make', 'makes',
    'give', 'gives', 'ensure', 'ensures', 'enable', 'enables',
    'allow', 'allows', 'provide', 'provides', 'deliver', 'delivers',
    'improve', 'improves', 'advance', 'advances', 'thinking',
    
    # ==================================================
    # TEMPORAL & SPATIAL TERMS
    # ==================================================
    'time', 'times', 'day', 'days', '2024', '2025', '2023',
    'way', 'ways',
    
    # ==================================================
    # ORGANIZATIONAL & STRUCTURAL NOUNS
    # ==================================================
    'group', 'groups', 'team', 'teams', 'people', 'role', 'roles', 'task', 'tasks',
    'strategy', 'strategies', 'model', 'models', 'program', 'programs',
    'initiative', 'initiatives', 'project', 'projects', 'goals', 'goal',
    'org', 'orgs', 'governance', 'access', 'productivity',
    
    # ==================================================
    # METADATA & DOCUMENTATION TERMS
    # ==================================================
    'blog', 'blogs', 'story', 'stories', 'article', 'articles',
    'page', 'pages', 'chapter', 'chapters', 'section', 'sections',
    'report', 'reports', 'paper', 'papers', 'guide', 'guides',
    'whitepaper', 'whitepapers',
    
    # ==================================================
    # ABSTRACT & GENERIC CONCEPTS
    # ==================================================
    'future', 'skill', 'skills',
    'language', 'languages', 'blueprint', 'blueprints', 'building',
    'cloud', 'clouds', 'governance',
    
    # ==================================================
    # CORPORATE/MANAGEMENT JARGON
    # ==================================================
    'context', 'understanding', 'conversation', 'conversations',
    'think', 'generative', 'capabilities', 'agentic',
    'adoption', 'empowered', 'higher', 'advanced',
    'activity', 'behaviors', 'usage',
    
    # ==================================================
    # CONTENT & PLATFORM SPECIFIC
    # ==================================================
    'youtube', 'content', 'ads', 'ad', 'grants', 'coding',
    'publishers', 'analysis',
}
stop_words.update(custom_stops)

# ==========================================
# PHASE 1: TEXT MINING & PRE-PROCESSING
# ==========================================
def clean_text(text):
    """Removes messy PDF formatting and line breaks."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_for_topics(text):
    return text  # keep raw text for embeddings

def mine_hierarchical_pdfs(base_dir):
    """Crawls the hierarchy, extracts text, and assigns metadata tags."""
    print(f"Starting extraction from: {base_dir}")
    all_data = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                
                # Extract metadata from folder paths
                path_parts = root.split(os.sep)
                level = path_parts[-2] if len(path_parts) > 2 else "Unknown_Level"
                company = path_parts[-1] if len(path_parts) > 1 else "Unknown_Company"
                
                try:
                    pdf_document = fitz.open(file_path)
                    full_text = ""
                    for page_num in range(len(pdf_document)):
                        full_text += pdf_document.load_page(page_num).get_text("text") + " "
                    
                    cleaned_text = clean_text(full_text)
                    processed_text = preprocess_for_topics(cleaned_text)
                    
                    all_data.append({
                        "Level": level,            
                        "Company": company,        
                        "FileName": file,
                        "Raw_Text": cleaned_text,  
                        "Processed_Text": processed_text 
                    })
                    pdf_document.close()
                    print(f"  [SUCCESS] Mined: {level} -> {company} -> {file}")
                    
                except Exception as e:
                    print(f"  [ERROR] Failed to mine {file}: {e}")
                    
    return pd.DataFrame(all_data)

# ==========================================
# PHASE 2: VISUALIZATION (KNOWLEDGE GRAPH)
# ==========================================
def generate_knowledge_graph(df, topic_model, output_dir):
    """Generates an interactive PyVis HTML network graph with:
    - Company → Topic edges (frequency)
    - Topic ↔ Topic edges (semantic similarity)
    - Topic ↔ Topic edges (co-occurrence via probabilities)
    """
    print("\nGenerating Interactive Knowledge Graph...")
    
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

    # ------------------------------------------
    # 1. Add Company Nodes
    # ------------------------------------------
    companies = df['Company'].unique()
    for company in companies:
        if company != "Unknown_Company":
            # Cast company to string to avoid JSON serialization errors
            net.add_node(str(company), label=str(company), color="#FFD700", size=50, shape="dot")

    # ------------------------------------------
    # 2. Add Topic Nodes FIRST (critical)
    # ------------------------------------------
    topic_info = topic_model.get_topic_info()
    valid_topics = topic_info[topic_info.Topic != -1]

    topic_ids = valid_topics['Topic'].tolist()

    for topic_id in topic_ids:
        words = topic_model.get_topic(topic_id)
        clean_name = " | ".join([word for word, _ in words[:3]])
        
        # Cast topic_id to int
        net.add_node(
            int(topic_id),
            label=clean_name,
            color="#87CEFA",
            size=30,
            shape="triangle"
        )

    # ------------------------------------------
    # 3. Topic–Topic Edges (Semantic Similarity)
    # ------------------------------------------
    from sklearn.metrics.pairwise import cosine_similarity

    topic_embeddings = topic_model.topic_embeddings_[valid_topics.index]
    similarity_matrix = cosine_similarity(topic_embeddings)

    threshold = 0.3  # tune if needed

    for i in range(len(topic_ids)):
        for j in range(i + 1, len(topic_ids)):
            sim = similarity_matrix[i][j]

            if sim > threshold:
                # Cast topic_ids to int, and sim to float
                net.add_edge(
                    int(topic_ids[i]),
                    int(topic_ids[j]),
                    value=float(sim * 5),
                    color="#555555",
                    title=f"Semantic similarity: {float(sim):.2f}"
                )

    # ------------------------------------------
    # 4. Topic–Topic Edges (Co-occurrence via probabilities)
    # ------------------------------------------
    if 'Topic_Distribution' in df.columns:
        co_occurrence_edges = set()  # Initialize the set to track unique edges
        
        for dist in df['Topic_Distribution']:
            top_topics = np.argsort(dist)[-3:]

            # Keep only meaningful probabilities
            strong_topics = [t for t in top_topics if dist[t] > 0.1]

            for i in range(len(strong_topics)):
                for j in range(i + 1, len(strong_topics)):
                    t1, t2 = int(strong_topics[i]), int(strong_topics[j])

                    if t1 != -1 and t2 != -1:
                        # Sort the tuple so (t1, t2) is treated the same as (t2, t1)
                        edge = tuple(sorted((t1, t2)))
                        
                        # Only add the edge if it hasn't been added yet
                        if edge not in co_occurrence_edges:
                            # t1 and t2 are already cast to int above
                            net.add_edge(
                                t1,
                                t2,
                                value=2,
                                color="#999999",
                                title="Co-occurrence (same document)"
                            )
                            co_occurrence_edges.add(edge)

    # ------------------------------------------
    # 5. Company–Topic Edges (Document Frequency)
    # ------------------------------------------
    df['Topic'] = topic_model.topics_
    valid_topics_df = df[df['Topic'] != -1]

    edges_data = (
        valid_topics_df
        .groupby(['Company', 'Topic'])
        .size()
        .reset_index(name='Weight')
    )

    for _, row in edges_data.iterrows():
        # Cast pandas outputs (numpy.int64) to native Python types (str, int)
        net.add_edge(
            str(row['Company']),
            int(row['Topic']),
            value=int(row['Weight']),
            title=f"Documents mapped: {int(row['Weight'])}"
        )

    # ------------------------------------------
    # 6. Layout & Save
    # ------------------------------------------
    net.repulsion(node_distance=300, spring_length=200)

    output_path = os.path.join(output_dir, "knowledge_graph_companies.html")
    net.write_html(output_path)

    print(f"  [SUCCESS] Knowledge Graph saved to: {output_path}")

# ==========================================
# PHASE 3: MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    data_directory = "./Thesis_Data_Mining" 
    output_dir = os.path.join(data_directory, "04_Analysis_Outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract Data
    df = mine_hierarchical_pdfs(data_directory)
    if df.empty:
        print("\nNo PDFs found. Check folder structure.")
        exit()

    df.to_csv(os.path.join(output_dir, "mined_corpus.csv"), index=False)

    # 2. BERTopic Modeling with KeyBERT and Entity Masking
    print("\nInitializing BERTopic Model with KeyBERT Representation and Entity Masking...")
    docs = df['Raw_Text'].tolist()
    classes_level = df['Level'].tolist()     
    classes_company = df['Company'].tolist() 
    
    umap_model = UMAP(
        n_neighbors=10, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )
    
    vectorizer_model = CountVectorizer(
        stop_words=list(stop_words),
        ngram_range=(1, 3)
    )
    
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    from hdbscan import HDBSCAN

    hdbscan_model = HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=3,
        calculate_probabilities=True,
        verbose=True,
        low_memory=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    df['Topic_Distribution'] = list(probs)
    
    # Save Topic details
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    print("\nTop Topics Discovered:")
    print(topic_info[['Topic', 'Count', 'Name']].head(5))

    # 3. Standard BERTopic Visualizations
    print("\nGenerating BERTopic HTML Visualizations...")
    
    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        fig_barchart.write_html(os.path.join(output_dir, "barchart_overall_topics.html"))
    except Exception as e:
        print(f"Could not generate overall barchart: {e}")
        
    try:
        topics_per_class_level = topic_model.topics_per_class(docs, classes=classes_level)
        fig_class_level = topic_model.visualize_topics_per_class(topics_per_class_level, top_n_topics=10)
        fig_class_level.write_html(os.path.join(output_dir, "topics_by_hierarchy_level.html"))
    except Exception as e:
        print(f"Could not generate level barchart: {e}")

    try:
        topics_per_class_company = topic_model.topics_per_class(docs, classes=classes_company)
        fig_class_comp = topic_model.visualize_topics_per_class(topics_per_class_company, top_n_topics=10)
        fig_class_comp.write_html(os.path.join(output_dir, "topics_by_company.html"))
    except Exception as e:
         print(f"Could not generate company barchart: {e}")

    # 4. Generate the PyVis Knowledge Graph
    generate_knowledge_graph(df, topic_model, output_dir)

    print(f"\nPipeline Complete! All data and interactive graphs saved to: {output_dir}")