"""
RAG pipeline orchestration for Cognitive RAG POC
"""
from typing import List, Dict, Any, Optional
import yaml
import os
from dotenv import load_dotenv
from .retriever import HybridRetriever
from .prompts import PromptManager
from .guardrails import GuardrailManager

# Load environment variables
load_dotenv()

class RAGChain:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        
        self.retriever = HybridRetriever(config_path)
        self.prompt_manager = PromptManager(config_path)
        self.guardrail_manager = GuardrailManager(config_path)
        
        self.prompt_fanning_enabled = self.config.get('prompt_fanning', {}).get('enabled', False)
        self.max_subqueries = self.config.get('prompt_fanning', {}).get('max_subqueries', 3)
    
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
        """Build retrieval index from chunks"""
        self.retriever.build_index(chunks)
        print("RAG chain index built successfully")
    
    def query(self, question: str, config_override: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main query processing pipeline
        """
        # Override config if provided
        if config_override:
            self._apply_config_override(config_override)
        
        try:
            # Step 1: Query decomposition (if enabled)
            if self.prompt_fanning_enabled:
                subqueries = self._decompose_query(question)
                retrieved_chunks = self._retrieve_multiple_queries(subqueries)
            else:
                retrieved_chunks = self._retrieve_single_query(question)
            
            # Step 2: Re-ranking
            reranking_applied = False
            top_k = self.config['retrieval']['top_k']
            rerank_top_k = self.config['retrieval']['rerank_top_k']
            
            print(f"Chain reranking check: {len(retrieved_chunks)} chunks, top_k: {top_k}, rerank_top_k: {rerank_top_k}")
            
            # Skip reranking if rerank_top_k equals top_k (naive RAG configuration)
            if rerank_top_k == top_k:
                print("Chain: Reranking disabled (rerank_top_k equals top_k)")
                retrieved_chunks = retrieved_chunks[:rerank_top_k]  # Just truncate to desired size
            elif len(retrieved_chunks) > rerank_top_k:
                print("Chain: Applying reranking...")
                retrieved_chunks = self.retriever.rerank(question, retrieved_chunks)
                reranking_applied = True
                print(f"Chain: Reranking applied, now have {len(retrieved_chunks)} chunks")
            else:
                print("Chain: No reranking needed")
            
            # Step 3: Answerability check (if enabled)
            answerability_enabled = self.config.get('guardrails', {}).get('answerability_enabled', True)
            if answerability_enabled:
                answerability_result = self.guardrail_manager.check_answerability(
                    question, retrieved_chunks
                )
                
                if not answerability_result['answerable']:
                    return {
                        "answer": "I don't have sufficient information to answer this question based on the available documents.",
                        "citations": [],
                        "confidence": 0.0,
                        "groundedness": "ungrounded",
                        "metadata": {
                            "answerability": answerability_result,
                            "retrieved_chunks": len(retrieved_chunks),
                            "techniques_used": self._get_techniques_used()
                        }
                    }
            else:
                # Skip answerability check for naive RAG
                answerability_result = {
                    "answerable": True,
                    "reasoning": "Answerability check disabled for naive RAG",
                    "available_info": "All retrieved chunks",
                    "missing_info": "None",
                    "confidence": None  # Explicitly set confidence to None for disabled answerability
                }
            
            # Step 4: Generate answer
            answer_result = self._generate_answer(question, retrieved_chunks)
            
            # Step 5: Self-check (if enabled)
            if self.config.get('guardrails', {}).get('self_check_enabled', False):
                self_check_result = self.guardrail_manager.self_check(
                    answer_result['answer'], retrieved_chunks
                )
                answer_result['self_check'] = self_check_result
                answer_result['groundedness'] = self_check_result['groundedness']
            
            # Step 6: Format response
            citations = self._format_citations(retrieved_chunks)
            
            # Apply dynamic citation scaling if enabled
            if self.config.get('guardrails', {}).get('dynamic_citations', False):
                citations = self._apply_dynamic_citations(citations, answerability_result.get('confidence'))
            
            response = {
                "answer": answer_result['answer'],
                "citations": citations,
                "groundedness": answer_result.get('groundedness', 'partially_grounded'),
                "metadata": {
                    "answerability": answerability_result,
                    "retrieved_chunks": len(retrieved_chunks),
                    "techniques_used": self._get_techniques_used(),
                    "self_check": answer_result.get('self_check', {}),
                    "processing_metrics": self._get_processing_metrics(question, retrieved_chunks, answerability_result, reranking_applied)
                }
            }
            
            # Always include confidence field (None for disabled answerability)
            response["confidence"] = answerability_result.get('confidence')
            
            return response
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "groundedness": "ungrounded",
                "metadata": {"error": str(e)}
            }
    
    def _decompose_query(self, question: str) -> List[str]:
        """Decompose query into sub-queries"""
        try:
            import openai
            
            subquery_types = self.config.get('prompt_fanning', {}).get('subquery_types', 
                ['definition', 'examples', 'latest_info'])
            
            prompt = f"""Break down this question into {self.max_subqueries} specific sub-queries to improve document retrieval:

Original question: {question}

Sub-query types to consider: {', '.join(subquery_types)}

Return {self.max_subqueries} specific sub-queries, one per line:
"""
            
            # Store the prompt for display
            self.decomposition_prompt = prompt
            
            response = openai.chat.completions.create(
                model=self.config['models']['llm_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            subqueries = [q.strip() for q in response.choices[0].message.content.split('\n') 
                         if q.strip()]
            
            # Add original question
            subqueries.insert(0, question)
            
            # Store the generated subqueries for display
            self.generated_subqueries = subqueries[:self.max_subqueries]
            
            return subqueries[:self.max_subqueries]
            
        except Exception as e:
            print(f"Error in query decomposition: {e}")
            return [question]
    
    def _retrieve_multiple_queries(self, subqueries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents for multiple sub-queries"""
        all_chunks = []
        chunk_ids = set()
        
        print(f"Multiple queries retrieval: {len(subqueries)} subqueries")
        
        for i, subquery in enumerate(subqueries):
            chunks = self.retriever.retrieve(subquery)
            print(f"Subquery {i+1}: Retrieved {len(chunks)} chunks")
            
            # Deduplicate by chunk ID
            for chunk in chunks:
                chunk_id = chunk['metadata']['chunk_id']
                if chunk_id not in chunk_ids:
                    all_chunks.append(chunk)
                    chunk_ids.add(chunk_id)
        
        print(f"After deduplication: {len(all_chunks)} unique chunks")
        
        # Sort by retrieval score
        all_chunks.sort(key=lambda x: x.get('retrieval_score', 0), reverse=True)
        
        return all_chunks
    
    def _retrieve_single_query(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve documents for single query"""
        chunks = self.retriever.retrieve(question)
        print(f"Single query retrieval: {len(chunks)} chunks")
        return chunks
    
    def _generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using retrieved chunks"""
        try:
            import openai
            
            # Get prompt template
            prompt_template = self.prompt_manager.get_answer_prompt()
            
            # Format chunks for prompt
            context = "\n\n".join([
                f"Document {i+1}:\n{chunk['content']}\nSource: {chunk['metadata']['source']}"
                for i, chunk in enumerate(chunks)
            ])
            
            # Fill prompt template
            prompt = prompt_template.format(
                question=question,
                context=context,
                max_citations=self.config.get('guardrails', {}).get('max_citations', 5)
            )
            
            # Store the prompt for display
            self.answer_prompt = prompt
            
            # Generate answer
            response = openai.chat.completions.create(
                model=self.config['models']['llm_model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('performance', {}).get('temperature', 0.1),
                max_tokens=self.config.get('performance', {}).get('max_tokens', 4000)
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "groundedness": "partially_grounded"  # Will be updated by self-check
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "groundedness": "ungrounded"
            }
    
    def _format_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format citations for response"""
        citations = []
        
        for chunk in chunks:
            citation = {
                "text": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content'],
                "source": chunk['metadata']['source'],
                "chunk_id": chunk['metadata']['chunk_id'],
                "score": chunk.get('retrieval_score', 0.0)
            }
            citations.append(citation)
        
        return citations
    
    def _get_techniques_used(self) -> List[str]:
        """Get list of techniques used in current configuration"""
        techniques = ["naive_rag"]
        
        if self.config['retrieval']['fusion_method'] != "dense_only":
            techniques.append("hybrid_retrieval")
        
        if self.prompt_fanning_enabled:
            techniques.append("prompt_fanning")
        
        if self.config.get('guardrails', {}).get('self_check_enabled', False):
            techniques.append("self_check")
        
        return techniques
    
    def _apply_config_override(self, config_override: Dict):
        """Apply configuration overrides"""
        for key, value in config_override.items():
            if key in self.config:
                self.config[key] = value
    
    def _apply_dynamic_citations(self, citations: List[Dict[str, Any]], confidence: float = None) -> List[Dict[str, Any]]:
        """Apply dynamic citation scaling based on confidence and quality"""
        if not citations:
            return citations
        
        # Base number of citations
        base_citations = self.config.get('guardrails', {}).get('max_citations', 5)
        
        # Scale based on confidence (higher confidence = more citations)
        if confidence is not None:
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        else:
            confidence_multiplier = 1.0  # Default when no confidence available
        
        # Scale based on citation quality (higher scores = more citations)
        if citations:
            avg_score = sum(c.get('score', 0) for c in citations) / len(citations)
            quality_multiplier = 0.7 + (avg_score * 0.3)  # Range: 0.7 to 1.0
        else:
            quality_multiplier = 1.0
        
        # Calculate dynamic citation count
        dynamic_count = int(base_citations * confidence_multiplier * quality_multiplier)
        dynamic_count = max(3, min(dynamic_count, len(citations)))  # Between 3 and all available
        
        # Sort by score and return top citations
        sorted_citations = sorted(citations, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_citations[:dynamic_count]
    
    def _get_processing_metrics(self, question: str, chunks: List[Dict[str, Any]], answerability_result: Dict, reranking_applied: bool = False) -> Dict[str, Any]:
        """Get actual processing metrics and data"""
        metrics = {
            "query_info": {
                "original_question": question,
                "question_length": len(question.split()),
                "question_type": self._classify_question_type(question),
                "complexity": self._assess_question_complexity(question)
            },
            "retrieval_metrics": {
                "chunks_retrieved": len(chunks),
                "retrieval_method": self.config['retrieval']['fusion_method'],
                "top_k": self.config['retrieval']['top_k'],
                "rerank_top_k": self.config['retrieval']['rerank_top_k'],
                "reranking_applied": reranking_applied,
                "reranking_method": "LLM" if reranking_applied else "None",
                "bm25_weight": self.config['retrieval'].get('bm25_weight', 0.0),
                "dense_weight": self.config['retrieval'].get('dense_weight', 1.0),
                "hybrid_retrieval": self.config['retrieval']['fusion_method'] != "dense_only",
                "total_index_chunks": len(self.retriever.chunks),
                "chunks_available_for_retrieval": len(self.retriever.chunks) >= self.config['retrieval']['top_k']
            },
            "prompt_fanning": {
                "enabled": self.prompt_fanning_enabled,
                "max_subqueries": self.max_subqueries if self.prompt_fanning_enabled else 0,
                "subqueries_generated": len(getattr(self, 'generated_subqueries', [])),
                "generated_subqueries": getattr(self, 'generated_subqueries', []),
                "decomposition_prompt": getattr(self, 'decomposition_prompt', None)
            },
            "guardrails": {
                "answerability_enabled": self.config.get('guardrails', {}).get('answerability_enabled', True),
                "answerability_threshold": self.config.get('guardrails', {}).get('answerability_threshold', 0.7),
                "self_check_enabled": self.config.get('guardrails', {}).get('self_check_enabled', False),
                "max_citations": self.config.get('guardrails', {}).get('max_citations', 5),
                "dynamic_citations": self.config.get('guardrails', {}).get('dynamic_citations', False)
            },
            "answerability_result": answerability_result,
            "chunk_scores": [
                {
                    "chunk_id": chunk.get('metadata', {}).get('chunk_id', f"chunk_{i}"),
                    "retrieval_score": chunk.get('retrieval_score', 0.0),
                    "source": chunk.get('metadata', {}).get('source', 'unknown'),
                    "content_length": len(chunk.get('content', ''))
                }
                for i, chunk in enumerate(chunks)
            ],
            "techniques_used": self._get_techniques_used(),
            "config_used": {
                "config_name": getattr(self, 'config_name', 'unknown'),
                "model": self.config['models']['llm_model'],
                "embedding_model": self.config['models']['embedding_model']
            },
            "prompts_used": {
                "answer_prompt": getattr(self, 'answer_prompt', None),
                "decomposition_prompt": getattr(self, 'decomposition_prompt', None)
            }
        }
        
        return metrics
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'definition']):
            return 'definition'
        elif any(word in question_lower for word in ['how', 'steps', 'process']):
            return 'procedural'
        elif any(word in question_lower for word in ['why', 'reason', 'because']):
            return 'explanatory'
        elif any(word in question_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            return 'locational'
        elif any(word in question_lower for word in ['list', 'examples', 'types']):
            return 'enumerative'
        else:
            return 'general'
    
    def _assess_question_complexity(self, question: str) -> str:
        """Assess the complexity of the question"""
        word_count = len(question.split())
        question_markers = question.count('?')
        conjunction_count = sum(1 for word in question.lower().split() if word in ['and', 'or', 'but', 'however'])
        
        if word_count > 20 or question_markers > 1 or conjunction_count > 2:
            return 'complex'
        elif word_count > 10 or conjunction_count > 1:
            return 'moderate'
        else:
            return 'simple'

# Example usage
if __name__ == "__main__":
    rag_chain = RAGChain()
    
    # Example query
    question = "What are the five pillars of the Well-Architected Framework?"
    
    # Process query
    result = rag_chain.query(question)
    
    print("Answer:", result['answer'])
    print("Groundedness:", result['groundedness'])
    print("Techniques used:", result['metadata']['techniques_used'])
