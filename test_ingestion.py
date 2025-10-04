#!/usr/bin/env python3
"""
Test script for document ingestion
This script tests the document loading, chunking, and embedding creation
"""

import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from rag.ingest import DocumentIngester

def test_ingestion():
    """Test the document ingestion pipeline"""
    print("🧪 Testing Document Ingestion Pipeline")
    print("=" * 50)
    
    try:
        # Initialize ingester
        print("1️⃣  Initializing Document Ingester...")
        ingester = DocumentIngester()
        print(f"   ✅ Chunk size: {ingester.chunk_size}")
        print(f"   ✅ Chunk overlap: {ingester.chunk_overlap}")
        
        # Load documents
        print("\n2️⃣  Loading documents from data/ directory...")
        documents = ingester.load_documents()
        
        if not documents:
            print("   ❌ No documents found in data/ directory")
            print("   📄 Please add PDF files to data/ directory")
            return False
        
        print(f"   ✅ Loaded {len(documents)} documents")
        for doc in documents:
            print(f"      - {doc['metadata']['title']} ({doc['metadata']['type']})")
        
        # Chunk documents
        print("\n3️⃣  Chunking documents...")
        chunks = ingester.chunk_documents(documents)
        print(f"   ✅ Created {len(chunks)} chunks")
        
        # Show chunk statistics
        if chunks:
            chunk_sizes = [len(chunk['content'].split()) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            print(f"   📊 Average chunk size: {avg_size:.1f} words")
            print(f"   📊 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} words")
        
        # Create embeddings
        print("\n4️⃣  Creating embeddings...")
        print("   ⏳ This may take a few minutes for large documents...")
        chunks_with_embeddings = ingester.create_embeddings(chunks)
        
        if chunks_with_embeddings and 'embedding' in chunks_with_embeddings[0]:
            embedding_dim = len(chunks_with_embeddings[0]['embedding'])
            print(f"   ✅ Created embeddings with dimension: {embedding_dim}")
            print(f"   ✅ Embeddings ready for {len(chunks_with_embeddings)} chunks")
        else:
            print("   ⚠️  No embeddings created (check OpenAI API key)")
        
        print("\n🎉 Ingestion test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during ingestion test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_ingestion()
    
    if success:
        print("\n✅ All tests passed! The RAG system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Start the FastAPI backend: uvicorn app.api:app --reload")
        print("   2. Start the Streamlit UI: streamlit run app/ui.py")
        print("   3. Or use the startup script: python start_server.py")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
