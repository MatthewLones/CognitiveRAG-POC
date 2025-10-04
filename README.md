# Cognitive RAG POC

A proof-of-concept RAG system demonstrating the evolution from Naïve RAG to enhanced techniques with measurable improvements.

## Problem Statement
Organizations can't safely query their proprietary data with raw LLMs. This POC shows how RAG can solve this with iterative improvements and clear technique references.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### 1. Setup Environment
```bash
# Clone and navigate to the repository
cd CognitiveRAG-POC

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Add Your Data
- Place PDFs in the `data/` directory
- The system supports PDFs, Markdown, and HTML files
- Sample: AWS Well-Architected Framework PDF is included

### 3. Test the System
```bash
# Test document ingestion
python test_ingestion.py

# Start the system (recommended)
python start_server.py

# Or start manually:
# Terminal 1: uvicorn app.api:app --reload
# Terminal 2: streamlit run app/ui.py
```

### 4. Access the Application
- **Streamlit UI**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Techniques Implemented

### Naïve RAG (Baseline)
- Retrieve-then-read with top-k dense retrieval
- Simple "stuff" prompt with citations

### Enhanced Techniques
1. **Hybrid Retrieval**: Sparse BM25 + dense embeddings with fusion
2. **Contextual Re-ranking**: LLM-based relevance scoring
3. **Prompt Fanning**: Parallel sub-queries with merge and deduplication
4. **Answerability + Abstain**: Routing/gating for weak evidence
5. **Evaluator-Optimizer**: Self-check verification against retrieved text

## Configuration

- `configs/naive.yaml`: Baseline Naïve RAG configuration
- `configs/naive_plus.yaml`: Enhanced RAG with all improvements
- `configs/base.yaml`: Common settings and parameters

## 🎯 Demo Script

### Recommended Demo Flow
1. **Start with Naïve RAG**: Ask "What is the Well-Architected Framework?"
   - Shows basic retrieval and simple citations
   
2. **Switch to Enhanced RAG**: Same question, better results
   - Hybrid retrieval finds more relevant chunks
   - Self-check improves groundedness
   
3. **Complex Query**: "How do I improve security and performance?"
   - Prompt fanning breaks down the query
   - Retrieves distinct sections for each aspect
   
4. **Compare Configurations**: Toggle between naïve and enhanced
   - Notice groundedness badge changes
   - See citation quality improvements

### Sample Queries
- "What are the five pillars of the Well-Architected Framework?"
- "How do I implement security best practices?"
- "What are the cost optimization strategies?"
- "How do I ensure reliability in my architecture?"
- "What performance optimization techniques are recommended?"

## Architecture

```
app/
├── api.py          # FastAPI endpoints
├── ui.py           # Streamlit interface
└── rag/
    ├── ingest.py   # Document loading and chunking
    ├── retriever.py # Hybrid retrieval system
    ├── chain.py    # RAG pipeline orchestration
    ├── prompts.py  # Prompt templates
    └── guardrails.py # Trust and safety measures
```

## Dataset
This POC uses the AWS Well-Architected Framework as the primary dataset, focusing on enterprise architecture best practices.

## 🔧 Troubleshooting

### Common Issues

**"Cannot connect to FastAPI server"**
- Make sure the backend is running: `uvicorn app.api:app --reload`
- Check if port 8000 is available

**"No documents found in data directory"**
- Add PDF files to the `data/` directory
- Run `python test_ingestion.py` to verify document loading

**"OpenAI API key not found"**
- Check your `.env` file has a valid `OPENAI_API_KEY`
- Ensure you have credits in your OpenAI account

**"Error creating embeddings"**
- Verify your OpenAI API key is correct
- Check your internet connection
- Ensure you have sufficient OpenAI credits

### Performance Tips
- First run will be slow as it creates embeddings
- Subsequent queries are much faster
- Use smaller chunk sizes for faster processing
- Enable caching for production use

## 📚 Key Libraries & References

- **FastAPI**: https://fastapi.tiangolo.com/ - Modern Python web framework
- **Streamlit**: https://streamlit.io/ - Data app framework
- **Sentence Transformers**: https://www.sbert.net/ - Text embeddings
- **FAISS**: https://faiss.ai/ - Vector similarity search
- **BM25**: https://github.com/dorianbrown/rank_bm25 - Sparse retrieval
- **PyPDF**: https://pypdf.readthedocs.io/ - PDF processing

## 🤝 Contributing
Built for QMIND AI club presentation. Focus on modular, measurable improvements to RAG systems.

## 📄 License
MIT License - Feel free to use and modify for your projects.
