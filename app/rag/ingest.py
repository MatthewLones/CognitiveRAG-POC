"""
Document ingestion and chunking for Cognitive RAG POC
"""
import os
from typing import List, Dict, Any
from pathlib import Path
import yaml

class DocumentIngester:
    """Handles document loading, chunking, and embedding storage"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        
        self.chunk_size = self.config['chunking']['chunk_size']
        self.chunk_overlap = self.config['chunking']['chunk_overlap']
        self.separators = self.config['chunking']['separators']
        self._embedding_model = None  # Cache the model
    
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
    
    def _get_embedding_model(self):
        """Get embedding model, loading it lazily if needed"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config['models']['embedding_model']
                print(f"Loading embedding model: {model_name}")
                self._embedding_model = SentenceTransformer(model_name)
                print("Embedding model loaded successfully")
            except ImportError:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
            except Exception as e:
                raise Exception(f"Error loading embedding model: {e}")
        return self._embedding_model
    
    def load_documents(self, data_dir: str = "data") -> List[Dict[str, Any]]:
        """
        Load documents from data directory
        Supports PDF, Markdown, and HTML files
        """
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Data directory {data_dir} not found. Creating...")
            data_path.mkdir(exist_ok=True)
            return documents
        
        # Load PDFs
        for pdf_file in data_path.glob("*.pdf"):
            doc = self._load_pdf(pdf_file)
            if doc:
                documents.append(doc)
        
        # Load Markdown files
        for md_file in data_path.glob("*.md"):
            doc = self._load_markdown(md_file)
            if doc:
                documents.append(doc)
        
        # Load HTML files
        for html_file in data_path.glob("*.html"):
            doc = self._load_html(html_file)
            if doc:
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents from {data_dir}")
        return documents
    
    def _load_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Load PDF document"""
        try:
            import pypdf
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return {
                    "content": text,
                    "metadata": {
                        "source": str(file_path),
                        "title": file_path.stem,
                        "type": "pdf",
                        "pages": len(pdf_reader.pages)
                    }
                }
        except ImportError:
            print("pypdf not installed. Install with: pip install pypdf")
            return None
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return None
    
    def _load_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Load Markdown document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                return {
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "title": file_path.stem,
                        "type": "markdown"
                    }
                }
        except Exception as e:
            print(f"Error loading Markdown {file_path}: {e}")
            return None
    
    def _load_html(self, file_path: Path) -> Dict[str, Any]:
        """Load HTML document"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                text = soup.get_text()
                
                return {
                    "content": text,
                    "metadata": {
                        "source": str(file_path),
                        "title": file_path.stem,
                        "type": "html"
                    }
                }
        except ImportError:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4")
            return None
        except Exception as e:
            print(f"Error loading HTML {file_path}: {e}")
            return None
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces with overlap
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_text(
                doc["content"], 
                doc["metadata"]
            )
            chunks.extend(doc_chunks)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using recursive splitting"""
        import tiktoken
        
        # Initialize tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Split text into tokens
        tokens = encoding.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = encoding.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk = {
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['source']}_{len(chunks)}",
                    "start_token": start_idx,
                    "end_token": end_idx,
                    "token_count": len(chunk_tokens)
                }
            }
            
            chunks.append(chunk)
            
            # Move start index with overlap
            start_idx = end_idx - self.chunk_overlap
            
            # Break if we've reached the end
            if end_idx >= len(tokens):
                break
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create embeddings for chunks
        """
        try:
            # Get cached model (loads lazily)
            model = self._get_embedding_model()
            
            # Extract texts
            texts = [chunk["content"] for chunk in chunks]
            
            # Create embeddings
            embeddings = model.encode(texts, show_progress_bar=True)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
            
            print(f"Created embeddings for {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return chunks

# Example usage
if __name__ == "__main__":
    ingester = DocumentIngester()
    
    # Load documents
    documents = ingester.load_documents()
    
    # Chunk documents
    chunks = ingester.chunk_documents(documents)
    
    # Create embeddings
    chunks_with_embeddings = ingester.create_embeddings(chunks)
    
    print(f"Processing complete. {len(chunks_with_embeddings)} chunks ready for retrieval.")
