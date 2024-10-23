import os
import sqlite3
from transformers import GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class WildRetrieval:
    def __init__(self, db_path='/apdcephfs_qy3/share_733425/timhuang/caiqi/long-short-uncertainty/factcheck_cache/wildhallu.db'):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            data = load_dataset("wentingzhao/WildHallucinations", split="train")
            self._initialize_database()
            self._tokenize_and_store(data)
        else:
            print('Database already exists. Skipping initialization.')
        
        self.encoder = None
        
        self.last_entity = None
        self.last_passages = None
        self.embed_cache = {}

    def _initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT,
                passage TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity ON passages (entity)')
        conn.commit()
        conn.close()

    def _tokenize_and_store(self, data):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in tqdm(data, desc='Tokenizing and storing data'):
            entity = item['entity']
            for webpage in item['info']:
                tokens = self.tokenizer.tokenize(webpage['text'])
                for i in range(0, len(tokens), 256):
                    passage = self.tokenizer.convert_tokens_to_string(tokens[i:i+256])
                    cursor.execute('INSERT INTO passages (entity, passage) VALUES (?, ?)', (entity, passage))
        conn.commit()
        conn.close()

    def load_encoder(self):
        encoder = SentenceTransformer("sentence-transformers/gtr-t5-large")
        encoder = encoder.cuda()
        encoder = encoder.eval()
        self.encoder = encoder

    def get_all_passages(self, entity):
        if self.last_entity == entity:
            print("Still using cached passages.")
            return self.last_passages
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT passage FROM passages WHERE entity = ?', (entity,))
        passages = [row[0] for row in cursor.fetchall()]
        conn.close()

        self.last_entity = entity
        self.last_passages = passages

        return passages

    def get_tfidf_passages(self, entity, query_sentence, top_k=5):
        passages = self.get_all_passages(entity)
        if not passages:
            return []

        vectorizer = TfidfVectorizer().fit_transform([query_sentence] + passages)
        vectors = vectorizer.toarray()

        query_vector = vectors[0]
        passage_vectors = vectors[1:]

        cosine_similarities = np.dot(passage_vectors, query_vector) / (np.linalg.norm(passage_vectors, axis=1) * np.linalg.norm(query_vector))
        top_indices = np.argsort(cosine_similarities)[::-1]

        return [passages[i] for i in top_indices[:top_k]]

    def get_bm25_passages(self, entity, query_sentence, top_k=5):
        passages = self.get_all_passages(entity)
        print("Passages retrieved: ", len(passages))
        if not passages:
            return []

        tokenized_passages = [self.tokenizer.tokenize(p) for p in passages]
        bm25 = BM25Okapi(tokenized_passages)
        tokenized_query = self.tokenizer.tokenize(query_sentence)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1]

        return [passages[i] for i in top_indices[:top_k]]

    def get_gtr_passages(self, entity, query_sentence, top_k=5):
        passages = self.get_all_passages(entity)
        print("Passages retrieved: ", len(passages))

        if not passages:
            return []

        if self.encoder is None:
            print("Loading encoder...")
            self.load_encoder()
        
        if entity in self.embed_cache:
            passage_embeddings = self.embed_cache[entity]
        else:
            passage_embeddings = self.encoder.encode(passages, device=self.encoder.device, batch_size=128)
            self.embed_cache[entity] = passage_embeddings
        
        query_embedding = self.encoder.encode([query_sentence], device=self.encoder.device, batch_size=128)[0]

        scores = np.inner(query_embedding, passage_embeddings)

        top_indices = np.argsort(scores)[::-1]

        return [passages[i] for i in top_indices[:top_k]]
    

# Example usage:
if __name__ == '__main__':

    retriever = WildRetrieval()

    entity = 'University of Cambridge'
    top_k = 2

    query_sentences = ['Cambridge University was founed in 1309.', "Cambridge University has 29 colleges."]
    # print("TF-IDF Relevant Passages:")
    # relevant_passages = retriever.get_tfidf_passages(entity, query_sentence, top_k)
    # for passage in relevant_passages:
    #     print(passage)
    
    # print("\nBM25 Relevant Passages:")
    # relevant_passages_bm25 = retriever.get_bm25_passages(entity, query_sentence, top_k)
    # for passage in relevant_passages_bm25:
    #     print(passage)
    print("\nGTR Relevant Passages:")
    for query_sentence in query_sentences:
        relevant_passages_gtr = retriever.get_gtr_passages(entity, query_sentence, top_k)
        for passage in relevant_passages_gtr:
            print(passage)
            print("%"*50)
