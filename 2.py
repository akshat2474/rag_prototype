import os
from dotenv import load_dotenv
import psutil
import gc
import time
import requests

load_dotenv()  # Load environment variables from .env file
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import json
import pandas as pd
from typing import List, Dict, Generator

class MacSafeRAG:
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb
        
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        
        print(f"✅ Model loaded. Current RAM: {self.get_memory_usage():.1f}GB")
    
    def get_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def check_memory_safe(self) -> bool:
        current = self.get_memory_usage()
        if current > self.max_memory_gb:
            print(f"⚠️ HIGH MEMORY: {current:.1f}GB / {self.max_memory_gb}GB")
            return False
        return True
    
    def stream_data(self, file_path: str, chunk_size: int = 300) -> Generator[tuple, None, None]:
        if file_path.endswith('.csv'):
            chunk_reader = pd.read_csv(file_path, chunksize=1000)
            chunk_id = 0
            
            for df_chunk in chunk_reader:
                for _, row in df_chunk.iterrows():
                    text = ' '.join([f"{col}: {val}" for col, val in row.items()])
                    
                    if len(text) > chunk_size:
                        for i, sub_chunk in enumerate([text[j:j+chunk_size] 
                                                     for j in range(0, len(text), chunk_size)]):
                            yield sub_chunk, {'source': file_path, 'chunk_id': f"{chunk_id}_{i}"}
                    else:
                        yield text, {'source': file_path, 'chunk_id': str(chunk_id)}
                    
                    chunk_id += 1
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                chunk_id = 0
                while True:
                    lines = []
                    for _ in range(1000):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                    
                    if not lines:
                        break
                    
                    text = ' '.join(lines)
                    if len(text) > chunk_size:
                        for i, sub_chunk in enumerate([text[j:j+chunk_size] 
                                                     for j in range(0, len(text), chunk_size)]):
                            yield sub_chunk, {'source': file_path, 'chunk_id': f"{chunk_id}_{i}"}
                    else:
                        yield text, {'source': file_path, 'chunk_id': str(chunk_id)}
                    
                    chunk_id += 1
    
    def process_safely(self, file_path: str, batch_size: int = 500):
        print(f"🚀 Starting safe processing of {file_path}")
        print(f"📊 Batch size: {batch_size}")
        
        batch_texts = []
        batch_metas = []
        total_processed = 0
        
        for text, metadata in self.stream_data(file_path):
            batch_texts.append(text)
            batch_metas.append(metadata)
            
            if len(batch_texts) >= batch_size:
                self._process_batch(batch_texts, batch_metas, total_processed)
                total_processed += len(batch_texts)
                
                batch_texts.clear()
                batch_metas.clear()
                gc.collect()
                
                current_mem = self.get_memory_usage()
                print(f"📈 Processed: {total_processed:,} chunks | RAM: {current_mem:.1f}GB")
                
                if not self.check_memory_safe():
                    print("⚠️ Memory getting high, taking a break...")
                    time.sleep(2)
                
                if current_mem > 14:
                    print("🛑 Memory critical - pausing 5 seconds")
                    time.sleep(5)
                    gc.collect()
        
        if batch_texts:
            self._process_batch(batch_texts, batch_metas, total_processed)
            total_processed += len(batch_texts)
        
        print(f"✅ Completed! Total chunks: {total_processed:,}")
        print(f"💾 Final RAM usage: {self.get_memory_usage():.1f}GB")
    
    def _process_batch(self, texts: List[str], metadatas: List[Dict], offset: int):
        embeddings = self.model.encode(texts, 
                                     batch_size=32,
                                     show_progress_bar=False)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings.astype('float32'))
        
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.metadata[offset + i] = {
                'text': text[:1000],
                'meta': meta
            }
    
    def save_index(self, save_path: str = "./mac_rag_index"):
        os.makedirs(save_path, exist_ok=True)
        
        faiss.write_index(self.index, f"{save_path}/faiss.index")
        
        with open(f"{save_path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"💾 Index saved to {save_path}")
        
        index_size = os.path.getsize(f"{save_path}/faiss.index") / (1024**2)
        meta_size = os.path.getsize(f"{save_path}/metadata.pkl") / (1024**2)
        
        print(f"📦 Index size: {index_size:.1f}MB")
        print(f"📦 Metadata size: {meta_size:.1f}MB")
    
    def load_index(self, save_path: str = "./mac_rag_index"):
        self.index = faiss.read_index(f"{save_path}/faiss.index")
        
        with open(f"{save_path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Index loaded from {save_path}")
        print(f"📊 Total vectors: {self.index.ntotal:,}")
    
    def query(self, question: str, k: int = 5) -> Dict:
        query_emb = self.model.encode([question])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype('float32'), k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.metadata:
                results.append({
                    'text': self.metadata[idx]['text'],
                    'score': float(score),
                    'metadata': self.metadata[idx]['meta']
                })
        
        context = "\n\n".join([
            f"Source {i+1} (score: {r['score']:.3f}):\n{r['text']}"
            for i, r in enumerate(results)
        ])
        
        return {
            'context': context,
            'results': results,
            'memory_usage': f"{self.get_memory_usage():.1f}GB"
        }

class OpenRouterClient:
    def __init__(self, api_key: str = None):
        # Get API key from environment variable or use provided key
        self.api_key = api_key or os.getenv("OPENROUTER_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def query_with_context(self, question: str, context: str, 
                        #   model: str = "meta-llama/llama-3.3-70b-instruct:free") -> str:
                        #   model: str = "google/gemma-2-9b-it:free") -> str:
                          model: str = "meta-llama/llama-3.1-405b-instruct:free") -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based on the provided context. Be accurate and concise."
            },
            {
                "role": "user", 
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Request failed: {str(e)}"

def main():
    """
    🎯 MAIN WORKFLOW - You only run this ONCE for setup!
    """
    
    # 📁 CHANGE THIS: Path to your CSV file
    CSV_FILE_PATH = "filtered_150k.csv"  # ← PUT YOUR CSV FILE PATH HERE
    
    # 📂 CHANGE THIS: Where to save the index (optional)
    INDEX_SAVE_PATH = "./my_rag_index"  # ← CHANGE IF YOU WANT
    
    # Get API key from environment variable
    OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
    
    print("🚀 Starting RAG System Setup...")
    print("This will take 10-20 minutes but you only do it ONCE!")
    
    # Step 1: Check if index already exists
    if os.path.exists(f"{INDEX_SAVE_PATH}/faiss.index"):
        print("✅ Found existing index! Skipping processing...")
        rag = MacSafeRAG()
        rag.load_index(INDEX_SAVE_PATH)
    else:
        # Step 2: Process your data (ONLY RUNS ONCE)
        print("📊 Processing your 150k lines...")
        rag = MacSafeRAG()
        rag.process_safely(CSV_FILE_PATH, batch_size=500)
        rag.save_index(INDEX_SAVE_PATH)
        print("🎉 Processing complete! Index saved.")
    
    # Step 3: Set up OpenRouter client
    openrouter = OpenRouterClient(OPENROUTER_KEY)
    
    # Step 4: Interactive query loop
    print("\n🤖 RAG System Ready! Ask questions (type 'quit' to exit):")
    
    while True:
        question = input("\n❓ Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q', 'close','stop']:
            print("Exiting✌🏼")
            break
        
        # Get relevant context from your data
        print("🔍 Searching your data...")
        rag_result = rag.query(question, k=5)
        
        # Send to OpenRouter with your 70B model
        print("🧠 Generating answer...")
        answer = openrouter.query_with_context(
            question, 
            rag_result['context'],
            model="meta-llama/llama-3.1-405b-instruct:free"  # Your 405B model!
            # model="meta-llama/llama-3.3-70b-instruct:free"  ## Your 70B model!
        )
        
        print(f"\n💬 Answer: {answer}")
        print(f"📊 Memory usage: {rag_result['memory_usage']}")

def quick_test():
    """
    🧪 TEST FUNCTION - Run this first to make sure everything works
    """
    print("🧪 Quick test mode...")
    
    # Test memory
    available = psutil.virtual_memory().available / (1024**3)
    print(f"Available RAM: {available:.1f}GB")
    
    if available < 4:
        print("❌ Not enough RAM available!")
        return
    
    # Test model loading
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test FAISS
    try:
        index = faiss.IndexFlatIP(384)
        print("✅ FAISS working")
    except Exception as e:
        print(f"❌ FAISS failed: {e}")
        return
    
    print("🎉 All tests passed! Ready to process your data.")

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 MAC-SAFE RAG SYSTEM")
    print("=" * 50)
    
    # Uncomment this line to run tests first:
    # quick_test()
    
    # Run the main system:
    main()

"""
🔧 WHAT TO CHANGE:

1. CSV_FILE_PATH = "your_data.csv"  ← Your CSV file path
2. OPENROUTER_KEY = "your_key_here" ← Your OpenRouter API key
3. INDEX_SAVE_PATH = "./my_rag_index" ← Where to save (optional)

🚀 HOW TO RUN:

First time (setup - takes 10-20 minutes):
python rag_system.py

Every other time (instant):
python rag_system.py  ← It loads the saved index!

📝 YOU NEVER RUN PROCESSING AGAIN!
Once the index is saved, it loads instantly every time.
"""