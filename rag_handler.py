import os
import gc
import time
import pickle
import psutil
import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Generator

class MacSafeRAG:
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb
        
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        
        print(f"Model loaded. Current RAM: {self.get_memory_usage():.1f}GB")
    
    def get_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    
    def check_memory_safe(self) -> bool:
        current = self.get_memory_usage()
        if current > self.max_memory_gb:
            print(f"HIGH MEMORY: {current:.1f}GB / {self.max_memory_gb}GB")
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
        print(f"Starting safe processing of {file_path}")
        print(f"Batch size: {batch_size}")
        
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
                    print("Memory critical - pausing 5 seconds")
                    time.sleep(5)
                    gc.collect()
        
        if batch_texts:
            self._process_batch(batch_texts, batch_metas, total_processed)
            total_processed += len(batch_texts)
        
        print(f"Completed! Total chunks: {total_processed:,}")
        print(f"Final RAM usage: {self.get_memory_usage():.1f}GB")
    
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
        
        print(f"Index saved to {save_path}")
        
        index_size = os.path.getsize(f"{save_path}/faiss.index") / (1024**2)
        meta_size = os.path.getsize(f"{save_path}/metadata.pkl") / (1024**2)
        
        print(f"Index size: {index_size:.1f}MB")
        print(f"Metadata size: {meta_size:.1f}MB")
    
    def load_index(self, save_path: str = "./mac_rag_index"):
        self.index = faiss.read_index(f"{save_path}/faiss.index")
        
        with open(f"{save_path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded from {save_path}")
        print(f"Total vectors: {self.index.ntotal:,}")
    
    def query(self, question: str, k: int = 5) -> Dict:
        start_time = time.time()
        query_emb = self.model.encode([question])[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        encode_time = time.time()
        print(f"  - Query encoding took: {encode_time - start_time:.2f}s")

        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype('float32'), k
        )
        search_time = time.time()
        print(f"  - FAISS search took: {search_time - encode_time:.2f}s")
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.metadata:
                results.append({
                    'text': self.metadata[idx]['text'],
                    'score': float(score),
                    'metadata': self.metadata[idx]['meta'],
                    'vector_id': int(idx),  # Add the vector index
                    'id': int(idx),  # Also add as 'id' for compatibility
                    'content': self.metadata[idx]['text'],  # Add as 'content' for compatibility
                    'source': self.metadata[idx]['meta']['source']  # Direct source access
                })
        
        context = "\n\n".join([
            f"Source {i+1} (score: {r['score']:.3f}, id: {r['vector_id']}):\n{r['text']}"
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
    
    def format_sources_text(self, results: List[Dict]) -> str:
        """Create a nicely formatted plain text list of sources."""
        if not results:
            return "\n\n---\n📚 No sources found"

        text_list = "\n\n---\nSources Used:\n"

        for i, result in enumerate(results, 1):
            source_file = os.path.basename(result['metadata']['source'])
            score = f"{result['score']:.3f}"
            vector_id = result.get('vector_id', result.get('id', 'Unknown'))
            preview = result['text'][:100].replace('\n', ' ').strip() + "..."
            
            text_list += f"\n[{i}] {source_file}\n"
            text_list += f"    Relevance: {score}\n"
            text_list += f"    Vector ID: {vector_id}\n"
            text_list += f"    Preview: \"{preview}\"\n"
        
        return text_list
    
    def query_with_context(self, question: str, context: str, results: List[Dict] = None,
                          model: str = "meta-llama/llama-3.1-405b-instruct:free") -> str:
        """Original method for regular Q&A with context"""
        
        system_message = """You are a helpful assistant that answers questions based on provided context sources. 
        
IMPORTANT INSTRUCTIONS:
- Answer based ONLY on the information provided in the context
- Always reference which source you're using (e.g., "According to Source 1..." or "Source 2 indicates...")
- If the context doesn't contain the answer, clearly state this
- Be accurate and specific
- Don't make assumptions beyond what's stated in the sources"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context from sources:\n{context}\n\nQuestion: {question}\n\nPlease answer based on the context above, citing your sources."}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            end_time = time.time()
            print(f"  - OpenRouter API call took: {end_time - start_time:.2f}s")
            
            response.raise_for_status() # Will raise an exception for 4xx/5xx errors
            
            answer = response.json()["choices"][0]["message"]["content"]
            
            if results:
                sources_text = self.format_sources_text(results)
                answer += sources_text
            
            return answer
        
        except requests.exceptions.RequestException as e:
            print(f"  - OpenRouter API call failed: {e}")
            return f"Error connecting to the AI model: {e}"
    
    def generate_graph_data(self, question: str, context: str, model: str, 
                           system_prompt: str, user_prompt: str) -> str:
        """Specialized method for generating graph data with custom prompts"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.0,  # Use temperature 0 for more deterministic JSON output
            "top_p": 1.0
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            end_time = time.time()
            print(f"  - OpenRouter graph API call took: {end_time - start_time:.2f}s")
            
            response.raise_for_status()
            
            answer = response.json()["choices"][0]["message"]["content"].strip()
            return answer
        
        except requests.exceptions.RequestException as e:
            print(f"  - OpenRouter graph API call failed: {e}")
            raise Exception(f"Failed to get graph data from OpenRouter: {e}")
        except KeyError as e:
            print(f"  - Unexpected response format: {e}")
            raise Exception(f"Unexpected response format from OpenRouter: {e}")
    
    def query_simple(self, question: str, model: str = "meta-llama/llama-3.1-405b-instruct:free") -> str:
        """Simple query without RAG context"""
        
        messages = [
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            end_time = time.time()
            print(f"  - OpenRouter simple API call took: {end_time - start_time:.2f}s")
            
            response.raise_for_status()
            
            answer = response.json()["choices"][0]["message"]["content"]
            return answer
        
        except requests.exceptions.RequestException as e:
            print(f"  - OpenRouter simple API call failed: {e}")
            return f"Error connecting to the AI model: {e}"