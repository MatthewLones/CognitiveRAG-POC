# Cognitive RAG POC

A proof-of-concept RAG system demonstrating the evolution from Naïve RAG to enhanced techniques with measurable improvements.

## Problem Statement
Organizations can't safely query their proprietary data with raw LLMs. This POC shows how RAG can solve this with iterative improvements and clear technique references.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your data**:
   - Place PDFs in the `data/` directory
   - The system supports PDFs, Markdown, and HTML files

3. **Run the application**:
   ```bash
   # Start the Streamlit UI
   streamlit run app/ui.py
   
   # Or use the FastAPI backend
   uvicorn app.api:app --reload
   ```

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

## Demo Script

1. **Easy lexical query**: Show hybrid retrieval beats dense-only
2. **Multi-facet query**: Demonstrate prompt fanning retrieves distinct sections
3. **Toggle self-check**: Show groundedness badge changes
4. **Performance comparison**: Display measurable improvements

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

## Contributing
Built for QMIND AI club presentation. Focus on modular, measurable improvements to RAG systems.
