from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json

model = SentenceTransformer('bert-base-nli-mean-tokens')

ALL_TOPICS = ['depression', 'anxiety', 'family-conflict']
TOPIC_EMBEDDING_FILE = 'topic-embeddings.json'

def calculate_topic_embeddings(topics):
    embeddings = {}
    for topic in topics:
        print(topic)
        df = pd.read_csv(f'data/meta_learn/{topic}.tsv', delimiter='\t', encoding='utf-8')
        unique_questions = df.iloc[:,0].unique()
        avg_embedding = model.encode(unique_questions).mean(0)
        embeddings[topic] = [float(v) for v in avg_embedding]
        print(embeddings[topic])
    with open(TOPIC_EMBEDDING_FILE, 'w') as f:
        json.dump(embeddings, f)

def read_topic_embeddings():
    with open(TOPIC_EMBEDDING_FILE, 'r') as f:
        embeddings = json.load(f)
    return embeddings

def find_topics(sentences, topic_embeddings):
    sentence_embeddings = model.encode(sentences)
    sim = cosine_similarity(
        sentence_embeddings,
        list(topic_embeddings.values())
    )  # (num_sents, num_topics)
    print(sim)

if __name__ == "__main__":
    sent = ['My mother controls my life.']
    # calculate_topic_embeddings(ALL_TOPICS)
    topic_embds = read_topic_embeddings()
    find_topics(sent, topic_embds)
