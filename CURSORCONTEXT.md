# Cognitive RAG POC - Cursor Context

## Project Overview
**Problem**: Organizations can't query their proprietary data with raw LLMs
**Solution**: Build a RAG solution with iterative improvements from Naïve RAG to enhanced techniques
**Goal**: POC for QMIND AI club presentation demonstrating RAG evolution with clear technique references

## 5-Hour Development Plan

### Hour 0 → 0.5: Bootstrap
- **Stack**: Python, LangChain/LlamaIndex, Chroma/FAISS, BM25, FastAPI + Streamlit, LiteLLM
- **Repo scaffold**: Set up directory structure with modular components

### Hour 0.5 → 1.5: Ingest + Baseline Naïve RAG
- Loaders: PDFs/MD/HTML with metadata extraction
- Chunking: Recursive 400-800 tokens, 60-100 token overlap
- Embeddings: text-embedding-3-large, persist vectors
- Baseline: Top-k dense retrieval → "stuff" prompt with citations

### Hour 1.5 → 2.5: RAG++ Enhancements
- **Hybrid retrieval**: Sparse BM25 + dense embeddings with RRF fusion
- **Re-ranking**: LLM re-rank top-20 to top-5 with justification
- **Prompt fanning**: Query decomposition into 2-4 sub-queries, parallel retrieval, dedupe, re-rank

### Hour 2.5 → 3.0: Guardrails for Trust
- **Answerability gate**: Threshold-based abstention with suggestions
- **Self-check pass**: Verify claims against cited chunks, drop ungrounded claims

### Hour 3.0 → 4.0: UI & Demo
- **Streamlit UI**: Config selector, question interface, citations, groundedness badges
- **Demo script**: Show hybrid vs dense, multi-facet queries, self-check improvements

### Hour 4.0 → 5.0: Documentation & Polish
- **README**: Setup instructions, techniques implemented, evaluation approach
- **Demo prep**: 3-minute presentation script with measurable improvements

## Techniques Implemented

### Naïve RAG (Baseline)
- Retrieve-then-read, top-k, stuff prompt with chunks + citations

### Enhanced Techniques
1. **Hybrid Retrieval**: Sparse BM25 + dense embeddings with fusion
2. **Contextual Re-ranking**: LLM-based relevance scoring
3. **Prompt Fanning**: Parallel sub-queries with merge and deduplication
4. **Answerability + Abstain**: Routing/gating for weak evidence
5. **Evaluator-Optimizer**: Self-check verification against retrieved text

## Repository Structure
```
cognitive-rag-poc/
├── app/
│   ├── api.py                 # FastAPI endpoints
│   ├── ui.py                  # Streamlit UI
│   └── rag/
│       ├── ingest.py          # loaders, chunking, embeddings
│       ├── retriever.py       # hybrid retrieval + fusion
│       ├── chain.py           # RAG pipeline (base + upgraded)
│       ├── prompts.py         # system/user templates
│       └── guardrails.py      # answerability + self-check
├── data/                      # AWS Well-Architected PDFs
├── configs/
│   ├── base.yaml             # k, chunk_size, models...
│   ├── naive.yaml
│   └── naive_plus.yaml
└── README.md
```

## Dataset
- **Primary**: AWS Well-Architected Framework PDFs
- **Focus**: Enterprise architecture best practices, security, reliability, performance

## Key Metrics to Track
- **Retrieval**: Hit@k, MRR, Context precision
- **Answer Quality**: Groundedness, Relevance, Completeness (LLM-as-judge)
- **Performance**: Latency, cost per query

## Development Approach
- **Modular**: Each technique as separate, composable component
- **Iterative**: Build baseline, then layer enhancements
- **Measurable**: Track improvements with clear metrics
- **Demo-ready**: Focus on presentation-worthy improvements

## Next Steps
1. Set up repository structure
2. Implement baseline Naïve RAG
3. Add hybrid retrieval
4. Implement re-ranking
5. Add prompt fanning
6. Build guardrails
7. Create Streamlit UI
8. Prepare demo script
