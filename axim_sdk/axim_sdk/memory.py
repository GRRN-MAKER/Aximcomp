import os
import json
import uuid
import math
from typing import List, Dict, Any, Optional

class AximMemory:
    """
    Axim Memory: A personalized, long-term memory layer built natively 
    without third-party wrappers like LiteLLM.
    
    It stores facts, preferences, and conversations, embedding them via Mistral 
    and saving them locally to persist across sessions.
    """
    def __init__(self, client, storage_path: str = "~/.axim_memory.json"):
        self.client = client
        self.storage_path = os.path.expanduser(storage_path)
        self.memories = self._load_memory()
        
    def _load_memory(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
        
    def _save_memory(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memories, f, indent=4)
        except Exception as e:
            print(f"Error saving Axim Memory: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        if not self.client:
            return []
        try:
            response = self.client.embeddings.create(model="mistral-embed", inputs=[text])
            return response.data[0].embedding
        except Exception:
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def add(self, text: str, user_id: str = "default_user", metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a new memory to the system.
        """
        embedding = self._get_embedding(text)
        if not embedding:
            return
            
        memory_obj = {
            "id": str(uuid.uuid4()),
            "text": text,
            "user_id": user_id,
            "metadata": metadata or {},
            "embedding": embedding
        }
        self.memories.append(memory_obj)
        self._save_memory()

    def search(self, query: str, user_id: str = "default_user", limit: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Searches for relevant past memories based on semantic similarity.
        """
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
            
        results = []
        for mem in self.memories:
            if mem.get("user_id") != user_id:
                continue
                
            sim = self._cosine_similarity(query_embedding, mem["embedding"])
            if sim >= threshold:
                results.append({"memory": mem, "score": sim})
                
        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return formatted top results
        return [r["memory"] for r in results[:limit]]
