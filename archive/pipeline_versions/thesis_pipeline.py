import os
import re
import pandas as pd
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from pyvis.network import Network
import numpy as np
import random
from umap import UMAP

# ==========================================
# SETUP: NLTK AND ADVANCED ENTITY MASKING
# ==========================================
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# THE SCIENTIFIC MASKING DICTIONARY
# Stripping brands, structural nouns, and generic filler to expose "Latent Values"
custom_stops = {
    # 1. Brand & Product Entities
    'microsoft', 'google', 'anthropic', 'claude', 'gemini', 'copilot', 'openai', 'chatgpt', 'azure', 'workspace', 'idc', 
    'youtube', 'googlers',
    
    # 2. Generic Educational Entities (Setting the "Scene")
    'education', 'higher', 'student', 'students', 'educator', 'educators', 'teacher', 'teachers', 
    'university', 'universities', 'college', 'colleges', 'campus', 'campuses', 'institutions', 'institution', 
    'school', 'schools', 'learning', 'teaching', 'academic', 'faculty', 'class', 'lesson', 
    'curriculum', 'classroom', 'learners', 'learn', 'knowledge', 'literacy',
    
    # 3. Generic Tech/Corporate Noise
    'ai', 'artificial', 'intelligence', 'technology', 'technologies', 'system', 'systems', 'platform', 
    'tool', 'tools', 'use', 'using', 'also', 'new', 'data', 'model', 'models', 'can', 'help', 'make',
    'report', 'business', 'support', 'usage', 'work', 'time', 'task', 'tasks', 'code', 'generative', 'user', 'users',
    
    # 4. Structural Noise & Web Scrape Leftovers
    'organization', 'organizations', 'across', 'like', 'blog', 'information', 'human', 'teams', 'conversations', 
    'educational', 'research', 'read', 'online', 'https', 'www', 'com', 'org', 'side', 'click', 'link', 'page', 'share',
    
    # 5. Numerical & Statistical (The "Accounting" noise)
    'billion', 'million', 'millions', 'numbers', 'percent', 'percentage', 'monthly', 'annual', 'year', 'years', 
    'increase', 'increased', 'total', 'index', 'activity', 'activities', 'growth', 'number', 'level',
    
    # 6. Administrative & Process "Fluff"
    'innovation', 'innovations', 'initiatives', 'initiative', 'blueprint', 'framework', 'approach', 'process', 
    'program', 'programs', 'project', 'projects', 'strategy', 'strategies', 'ready', 'readiness',
    'opportunity', 'opportunities', 'potential', 'impact', 'benefits', 'successful', 'success',
    'foundational', 'characteristics', 'advanced', 'future', 'modern', 'effective', 'transformative',
    'insights', 'development', 'developing', 'outcomes', 'results', 'helped', 'progress', 'improve', 
    'plan', 'resources', 'ensuring', 'experts', 'collaboration', 'based', 'management', 'leaders',
    'discover', 'create', 'applications', 'challenges', 'principles', 'policies', 'policy', 'leading',
    'launch', 'staff', 'drive', 'supported', 'change', 'enhance', 'access', 'foundation',
    
    # 7. Document & Research Artifacts
    'appendix', 'contents', 'table', 'figure', 'screenshot', 'participants', 'survey', 
    'respondents', 'study', 'empirical', 'findings', 'conclusion', 'notes', 'brief',
    'explore', 'excited', 'share', 'patterns', 'example', 'examples', 'case', 'specifically',
    'sponsored',
    
    # 8. Modal & High-Frequency Verbs/Adverbs
    'working', 'within', 'include', 'including', 'provide', 'providing', 'enable', 
    'enabling', 'focus', 'focused', 'become', 'set', 'well', 'way', 'ways', 'looking', 
    'forward', 'participation', 'three', 'problem', 'helping', 'come', 'first',
    
    # 9. Economics, Industry & Platform Noise (The "Topic 2" Scrub)
    'economic', 'economical', 'economy', 'businesses', 'companies', 'industry', 'industries', 'investment', 
    'cost', 'cloud', 'services', 'products', 'global', 'national', 'state', 'states', 'america',
    'ads', 'ad', 'advertising', 'creators', 'publishers', 'marketing', 'market', 'developers',
    'grants', 'nonprofits', 'worth', 'donated', 'search', 'philanthropic', 'reports', 'customers',
    'employee', 'customer', 'cash', 'scale', 'requests', 'built', 'funding', 'internet', 'employees', 'revenue',
    
    # 10. Functional & Technical Descriptions
    'assistant', 'assistance', 'capabilities', 'interaction', 'interactions', 'coding', 'tech', 
    'agentic', 'agents', 'automated', 'automation', 'skills', 'skill', 'training',
    'api', 'software', 'design', 'actions', 'writing', 'content', 'language', 'genai', 'concepts', 'ml',
    
    # 11. Speakers & Events
    'ben', 'gomes', 'speaker', 'speech', 'transcript', 'forum', 'event', 'summit', 'leadership', 'talk',
    
    # 12. Strategic Slogans & Directional Filler
    'empowered', 'power', 'possible', 'toward', 'goals', 'move', 'elevate', 'building', 'build', 'strategic', 
    'guidance', 'exploring', 'explore', 'value', 'values', 'governance',
    
    # 13. PR & Government Artifacts (The "Topic -1" Scrub)
    'af', 'challenge','youth', 'pledge', 'taskforce', 'commitments',
    
    # 14. Behavioral & Research Noise
    'et', 'al', 'per', 'analyze', 'analyzing', 'analysis', 'understanding', 'behavior', 
    'behaviors', 'behavioral', 'conversation', 'conversations', 'responsibly',
    
    # 15. The "Values-Only" Extraction Filter
    'infrastructure', 'infrastructural', 'expertise', 'operations', 'operational', 
    'scientific', 'maturity', 'making', 'processes', 
    'complex', 'queries', 'questions', 'creating', 'organizational', 
    'develop', 'analytics', 'roi', 'technical', 'company', 'implementing', 
    'service', 'product', 'engineers', 'primitives', 'related', 'document', 
    'context', 'scaling', 'institutional', 'next', 'practices', 'introducing', 'prompt', 
    'feedback', 'sector', 'driven', 'donations', 'significant', 'energy', 
    'digital', 'transformation', 'key', 'needs', 'estimate', 'thinking',
    'self', 'analyzing', 'analysis', 'practices', 'sectors', 'highly', 'phone', 
    'deliver', 'maps', 'shift', 'driver', 'response', 'tutoring', 'prepare',
    'developed', 'education', 'provides', 'evaluations',
    'manage', 'spark', 'machine', 'adoption', 'said', 'communities', 'ensure',
    'scientists', 'critical', 'important', 'constitution', 'faster', 'powered', 
    'establish', 'community', 'expand', 'ideas', 'researchers', 'savings', 
    'already', 'many', 'journey', 'administrative', 'continue', 'guide', 
    'engage', 'teaming', 'things', 'core', 'people', 'names',
    'day', 'connections', 'giving', 'thousands', 'tens', 'house', 'signing', 
    'presidential', 'joins', 'pledge', 'taskforce', 'picoctf', 'cybersecurity', 
    'ed', 'experience', 'volunteered', 'production', 'teach', 'rwandan', 
    'instructors', 'professor', 'professors', 'tel', 'aviv', 'makers', 'auburn', 
    'us', 'kids', 'course', 'courses',
    # 'productivity', 'security', 'science', 'risk', 'risks'
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
    """Initial regex cleaning (Stopwords are handled by CountVectorizer later)."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return " ".join(words)

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
    """Generates an interactive PyVis HTML network graph."""
    print("\nGenerating Interactive Knowledge Graph...")
    
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # Add Company Nodes 
    companies = df['Company'].unique()
    for company in companies:
        if company != "Unknown_Company":
            net.add_node(company, label=company, color="#FFD700", size=50, shape="dot")

    # Map topics back to dataframe
    df['Topic'] = topic_model.topics_
    valid_topics_df = df[df['Topic'] != -1] # Exclude noise bin
    
    topic_info = topic_model.get_topic_info()
    topic_mapping = dict(zip(topic_info['Topic'], topic_info['Name']))

    edges_data = valid_topics_df.groupby(['Company', 'Topic']).size().reset_index(name='Weight')

    # Add Topic Nodes and Draw Edges
    added_topics = set()
    for index, row in edges_data.iterrows():
        company = row['Company']
        topic_id = row['Topic']
        weight = row['Weight']
        
        # Clean topic label (Extracting the top 3 KeyBERT words)
        raw_name = topic_mapping[topic_id]
        clean_name = " | ".join(raw_name.split("_")[1:4]) 
        
        if topic_id not in added_topics:
            net.add_node(topic_id, label=clean_name, color="#87CEFA", size=30, shape="triangle")
            added_topics.add(topic_id)
            
        # Draw Edge
        net.add_edge(company, topic_id, value=weight, title=f"Documents mapping to this topic: {weight}")

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
    docs = df['Processed_Text'].tolist()
    classes_level = df['Level'].tolist()     
    classes_company = df['Company'].tolist() 

    # The Scientific Upgrades: N-Grams and KeyBERT
    vectorizer_model = CountVectorizer(stop_words=list(stop_words), ngram_range=(1, 3))
    representation_model = KeyBERTInspired()
    
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42 # <--- This freezes the map shape
    )

    topic_model = BERTopic(
        language="english", 
        calculate_probabilities=True, 
        verbose=True, 
        min_topic_size=3, 
        nr_topics="auto",
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        umap_model=umap_model
    )
    
    topics, probs = topic_model.fit_transform(docs)
    
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