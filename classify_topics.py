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

def topic_embeddings_from_words(topics):
    embds = model.encode(topics)
    return {topic: embds[i] for i, topic in enumerate(topics)}

def find_topics(sentences, topic_embeddings):
    sentence_embeddings = model.encode(sentences)
    sim = cosine_similarity(
        sentence_embeddings,
        list(topic_embeddings.values())
    )  # (num_sents, num_topics)
    print(sim)

if __name__ == "__main__":
    sent = ['My husband and I have been together for seven years now.']
    sent = ["I’m a teenager. My entire family needs family therapy, and more than likely individual therapy. My parents refuse to take action, and I'm tired of it. Is there any way I can get out of this myself?"]
    sent = ["When I'm in large crowds I get angry and I just can't deal with people. I don't really like other people (I prefer animals) they make me nervous and scared. I lay awake at night thinking and having conversations in my head and i almost always end up making myself feel terrible and crying, I have more conversions in my head than I do with actual people."]
    # calculate_topic_embeddings(ALL_TOPICS)
    topic_embds = read_topic_embeddings()
    # topic_embds = topic_embeddings_from_words(ALL_TOPICS)
    find_topics(sent, topic_embds)
