"""
Hybrid retrieval system for Cognitive RAG POC
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HybridRetriever:
    """Implements hybrid retrieval with BM25 + Dense embeddings + Re-ranking"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        
        self.top_k = self.config['retrieval']['top_k']
        self.rerank_top_k = self.config['retrieval']['rerank_top_k']
        self.fusion_method = self.config['retrieval']['fusion_method']
        self.bm25_weight = self.config['retrieval']['bm25_weight']
        self.dense_weight = self.config['retrieval']['dense_weight']
        
        self.chunks = []
        self.dense_index = None
        self.bm25_index = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with support for extends"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle extends directive
        if 'extends' in config:
            base_path = f"configs/{config['extends']}"
            base_config = self._load_config(base_path)
            
            # Merge configs (child overrides parent)
            merged_config = base_config.copy()
            for key, value in config.items():
                if key != 'extends':
                    if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                        merged_config[key] = {**merged_config[key], **value}
                    else:
                        merged_config[key] = value
            
            return merged_config
        
        return config
        
    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build both BM25 and dense vector indices"""
        self.chunks = chunks
        
        # Build BM25 index
        self._build_bm25_index()
        
        # Build dense vector index
        self._build_dense_index()
        
        print(f"Built indices for {len(chunks)} chunks")
        self._print_index_stats()
    
    def _print_index_stats(self):
        """Print index statistics for debugging"""
        print(f"📊 Index Statistics:")
        print(f"   Total chunks: {len(self.chunks)}")
        print(f"   BM25 index: {'✅ Built' if self.bm25_index is not None else '❌ Failed'}")
        print(f"   Dense index: {'✅ Built' if self.dense_index is not None else '❌ Failed'}")
        print(f"   Config top_k: {self.top_k}")
        print(f"   Config rerank_top_k: {self.rerank_top_k}")
        
        if len(self.chunks) < self.top_k:
            print(f"⚠️  WARNING: Index only has {len(self.chunks)} chunks but top_k is {self.top_k}")
            print(f"   This means you can only retrieve {len(self.chunks)} chunks maximum")
        else:
            print(f"✅ Index has sufficient chunks for top_k={self.top_k} retrieval")
    
    def _build_bm25_index(self):
        """Build BM25 sparse retrieval index"""
        try:
            from rank_bm25 import BM25Okapi
            
            # Extract texts for BM25
            texts = [chunk["content"] for chunk in self.chunks]
            
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_texts)
            
            print("BM25 index built successfully")
            
        except ImportError:
            print("rank_bm25 not installed. Install with: pip install rank-bm25")
            self.bm25_index = None
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            self.bm25_index = None
    
    def _build_dense_index(self):
        """Build dense vector index using FAISS"""
        try:
            import faiss
            
            # Extract embeddings and determine dimension
            embeddings = []
            embedding_dim = None
            print("Building dense index...")
            
            for chunk in self.chunks:
                if "embedding" in chunk and chunk["embedding"]:
                    embedding = chunk["embedding"]
                    embeddings.append(embedding)
                    if embedding_dim is None:
                        embedding_dim = len(embedding)
                else:
                    # Skip chunks without embeddings for now
                    continue
            
            if not embeddings:
                print("No embeddings found in chunks")
                self.dense_index = None
                return
                
            # Use actual embedding dimension
            if embedding_dim is None:
                # Try to get dimension from config or default to 384
                try:
                    from sentence_transformers import SentenceTransformer
                    model_name = self.config['models']['embedding_model']
                    model = SentenceTransformer(model_name)
                    embedding_dim = model.get_sentence_embedding_dimension()
                except:
                    embedding_dim = 384  # Fallback for all-MiniLM-L6-v2

            print("Building dense index... with embedding dimension:", embedding_dim)
            
            
            # Convert to numpy array
            embeddings = np.array(embeddings).astype('float32')
            print(f"Converted embeddings to numpy array: shape={embeddings.shape}, dtype={embeddings.dtype}")
            
            # Build FAISS index with dynamic dimension
            self.dense_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            print("FAISS index created successfully")
            
            # Manual L2 normalization (avoiding FAISS normalize_L2 bug on Mac M2)
            print("Performing manual L2 normalization... (This is because the FAISS normalize_L2 bug on Mac M2)")
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            print("Manual L2 normalization completed")
            
            # Add embeddings to index
            print("Adding embeddings to FAISS index...")
            self.dense_index.add(embeddings)
            print("Embeddings added to index successfully")
            
            print("Dense vector index built successfully")
            
        except ImportError:
            print("faiss-cpu not installed. Install with: pip install faiss-cpu")
            self.dense_index = None
        except Exception as e:
            print(f"Error building dense index: {e}")
            self.dense_index = None
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval
        """
        if top_k is None:
            top_k = self.top_k
        
        print(f"Retriever: Retrieving with top_k={top_k}, fusion_method={self.fusion_method}")
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, top_k)
        print(f"Retriever: BM25 returned {len(bm25_results)} results")
        
        # Get dense results
        dense_results = self._dense_search(query, top_k)
        print(f"Retriever: Dense returned {len(dense_results)} results")
        
        # Fuse results
        fused_results = self._fuse_results(bm25_results, dense_results, top_k)
        print(f"Retriever: Fused results: {len(fused_results)} chunks")
        
        # Verification: Check if we got the expected number of chunks
        if len(fused_results) < top_k:
            print(f"⚠️  WARNING: Expected {top_k} chunks but only got {len(fused_results)}")
            print(f"   This could be due to:")
            print(f"   - Index only contains {len(self.chunks)} total chunks")
            print(f"   - BM25 index issues: {len(bm25_results)} results")
            print(f"   - Dense index issues: {len(dense_results)} results")
        else:
            print(f"✅ Successfully retrieved {len(fused_results)} chunks (expected {top_k})")
        
        return fused_results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform BM25 search"""
        if self.bm25_index is None:
            return []
        
        try:
            # Tokenize query
            query_tokens = query.split()

            print(f"BM25 Searching")
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
            
        except Exception as e:
            print(f"Error in BM25 search: {e}")
            return []
    
    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform dense vector search"""
        if self.dense_index is None:
            return []
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load embedding model
            model_name = self.config['models']['embedding_model']
            model = SentenceTransformer(model_name)
            
            # Create query embedding
            query_embedding = model.encode([query])
            
            # Manual normalization for query embedding (matching index normalization)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # Search
            scores, indices = self.dense_index.search(query_embedding, top_k)
            
            # Return results
            return [(idx, float(score)) for idx, score in zip(indices[0], scores[0])]
            
        except Exception as e:
            print(f"Error in dense search: {e}")
            return []
    
    def _fuse_results(self, bm25_results: List[Tuple[int, float]], 
                     dense_results: List[Tuple[int, float]], 
                     top_k: int) -> List[Dict[str, Any]]:
        """Fuse BM25 and dense results"""
        
        if self.fusion_method == "dense_only":
            # Return only dense results
            results = dense_results
        elif self.fusion_method == "bm25_only":
            # Return only BM25 results
            results = bm25_results
        else:
            # Reciprocal Rank Fusion (RRF)
            results = self._reciprocal_rank_fusion(bm25_results, dense_results)
        
        # Convert to final format
        final_results = []
        for idx, score in results[:top_k]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["retrieval_score"] = score
                final_results.append(chunk)
        
        return final_results
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Tuple[int, float]], 
                               dense_results: List[Tuple[int, float]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """Implement Reciprocal Rank Fusion"""
        
        # Create score dictionaries
        bm25_scores = {idx: score for idx, score in bm25_results}
        dense_scores = {idx: score for idx, score in dense_results}
        
        # Get all unique indices
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for idx in all_indices:
            rrf_score = 0.0
            
            # BM25 contribution
            if idx in bm25_scores:
                bm25_rank = next(i for i, (iidx, _) in enumerate(bm25_results) if iidx == idx) + 1
                rrf_score += self.bm25_weight / (k + bm25_rank)
            
            # Dense contribution
            if idx in dense_scores:
                dense_rank = next(i for i, (iidx, _) in enumerate(dense_results) if iidx == idx) + 1
                rrf_score += self.dense_weight / (k + dense_rank)
            
            rrf_scores[idx] = rrf_score
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using LLM or cross-encoder
        """
        print(f"Reranking: {len(chunks)} chunks, rerank_top_k: {self.rerank_top_k}")
        
        if len(chunks) <= self.rerank_top_k:
            print(f"No reranking needed: {len(chunks)} <= {self.rerank_top_k}")
            return chunks
        
        print(f"Applying LLM reranking to {len(chunks)} chunks...")
        try:
            # Use LLM for re-ranking
            reranked_chunks = self._llm_rerank(query, chunks)
            print(f"LLM reranking completed, returning top {self.rerank_top_k} chunks")
            return reranked_chunks[:self.rerank_top_k]
            
        except Exception as e:
            print(f"Error in re-ranking: {e}")
            return chunks[:self.rerank_top_k]
    
    def _llm_rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"LLM re-ranking...")
        """Re-rank using LLM"""
        try:
            import openai
            
            # Create prompt for re-ranking
            prompt = f"""You are a document retrieval expert. Given a query and a list of document chunks, rank them by relevance to the query.

Query: {query}

Document chunks:
"""
            
            for i, chunk in enumerate(chunks):
                prompt += f"\n{i+1}. {chunk['content'][:200]}...\n"
            
            prompt += f"""
Please rank these {len(chunks)} chunks by relevance to the query. Return only the numbers in order of relevance, separated by commas.

Example: 3,1,5,2,4
"""
            
            # Call LLM
            response = openai.chat.completions.create(
                model=self.config['models']['llm_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            
            # Parse response
            ranking_text = response.choices[0].message.content.strip()
            ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
            
            # Reorder chunks
            reranked = [chunks[i] for i in ranking if i < len(chunks)]
            
            return reranked
            
        except Exception as e:
            print(f"Error in LLM re-ranking: {e}")
            return chunks
    
    def test_retrieval(self, query: str = "test query"):
        """Test method to verify retrieval is working correctly"""
        print(f"\n🧪 Testing retrieval with query: '{query}'")
        print(f"   Requesting {self.top_k} chunks...")
        
        results = self.retrieve(query)
        
        print(f"   Results: {len(results)} chunks retrieved")
        print(f"   Expected: {self.top_k} chunks")
        
        if len(results) == self.top_k:
            print(f"   ✅ SUCCESS: Retrieved exactly {self.top_k} chunks")
        elif len(results) < self.top_k:
            print(f"   ⚠️  WARNING: Only retrieved {len(results)} chunks (expected {self.top_k})")
        else:
            print(f"   ❓ UNEXPECTED: Retrieved {len(results)} chunks (expected {self.top_k})")
        
        return results

# Example usage
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    # Example chunks (would normally come from ingest.py)
    example_chunks = [
        {
            "content": "The Well-Architected Framework provides guidance to help you build secure, high-performing, resilient, and efficient infrastructure for your applications.",
            "metadata": {"source": "test.pdf", "chunk_id": "0"},
            "embedding": [0.1] * 384  # all-MiniLM-L6-v2 dimension
        }
    ]
    
    # Build index
    retriever.build_index(example_chunks)
    
    # Retrieve
    results = retriever.retrieve("What is the Well-Architected Framework?")
    print(f"Retrieved {len(results)} chunks")
