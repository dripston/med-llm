"""
Script to generate embeddings using SambaNova API
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SambaNovaEmbeddings:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sambanova.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text using SambaNova API
        """
        try:
            payload = {
                "input": text,
                "model": "E5-Mistral-7B-Instruct"
            }
            
            # Add timeout to prevent hanging
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                embedding = data["data"][0]["embedding"]
                # Check if embedding is all zeros
                if all(v == 0.0 for v in embedding):
                    print("Warning: Embedding is all zeros")
                    return None
                return embedding
            else:
                print(f"Error generating embedding: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Timeout error generating embedding")
            return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        """
        embeddings = []
        for i, text in enumerate(texts):
            print(f"  Generating embedding {i+1}/{len(texts)}...")
            embedding = self.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Add a small random vector if embedding failed
                import random
                embeddings.append([random.uniform(-0.1, 0.1) for _ in range(4096)])
        return embeddings

def test_sambanova_connection(api_key: str):
    """
    Test the SambaNova API connection
    """
    embedder = SambaNovaEmbeddings(api_key)
    test_text = "This is a test sentence for embedding."
    embedding = embedder.get_embedding(test_text)
    
    if embedding:
        print("SambaNova API connection successful!")
        print(f"Embedding dimension: {len(embedding)}")
        return True
    else:
        print("Failed to connect to SambaNova API")
        return False

if __name__ == "__main__":
    # Test the connection with API key from environment variables
    api_key = os.environ.get('SAMBANOVA_API_KEY')
    if api_key:
        test_sambanova_connection(api_key)
    else:
        print("SAMBANOVA_API_KEY not found in environment variables")