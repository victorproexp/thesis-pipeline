import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

csv_path = "Thesis_Data_Mining/04_Analysis_Outputs/corpus.csv"
df = pd.read_csv(csv_path)

documents = df['Processed_Text'].tolist()
print(f"Loaded {len(documents)} documents")

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

topic_model = BERTopic(
    embedding_model=embedding_model,
    embedding_batch_size=32,
    language="english",
    calculate_probabilities=True,
    min_topic_size=3
)

topics, probs = topic_model.fit_transform(documents)
print(f"Fitted {len(topics)} topics")

topic_info = topic_model.get_topic_info()
print(f"\nTopic info columns: {topic_info.columns.tolist()}")

print(f"\nFirst 5 rows of Representative_Docs:")
for i in range(min(5, len(topic_info))):
    val = topic_info['Representative_Docs'].iloc[i]
    print(f"\nRow {i}:")
    print(f"  Type: {type(val)}")
    print(f"  Value: {val}")
