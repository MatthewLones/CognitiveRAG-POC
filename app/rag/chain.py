"""
RAG pipeline orchestration for Cognitive RAG POC
"""
from typing import List, Dict, Any, Optional
import yaml
from .retriever import HybridRetriever
from .prompts import PromptManager
from .guardrails import GuardrailManager

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
            if len(retrieved_chunks) > self.config['retrieval']['rerank_top_k']:
                retrieved_chunks = self.retriever.rerank(question, retrieved_chunks)
            
            # Step 3: Answerability check
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
            response = {
                "answer": answer_result['answer'],
                "citations": self._format_citations(retrieved_chunks),
                "confidence": answerability_result['confidence'],
                "groundedness": answer_result.get('groundedness', 'partially_grounded'),
                "metadata": {
                    "answerability": answerability_result,
                    "retrieved_chunks": len(retrieved_chunks),
                    "techniques_used": self._get_techniques_used(),
                    "self_check": answer_result.get('self_check', {})
                }
            }
            
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
            
            return subqueries[:self.max_subqueries]
            
        except Exception as e:
            print(f"Error in query decomposition: {e}")
            return [question]
    
    def _retrieve_multiple_queries(self, subqueries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents for multiple sub-queries"""
        all_chunks = []
        chunk_ids = set()
        
        for subquery in subqueries:
            chunks = self.retriever.retrieve(subquery)
            
            # Deduplicate by chunk ID
            for chunk in chunks:
                chunk_id = chunk['metadata']['chunk_id']
                if chunk_id not in chunk_ids:
                    all_chunks.append(chunk)
                    chunk_ids.add(chunk_id)
        
        # Sort by retrieval score
        all_chunks.sort(key=lambda x: x.get('retrieval_score', 0), reverse=True)
        
        return all_chunks
    
    def _retrieve_single_query(self, question: str) -> List[Dict[str, Any]]:
        """Retrieve documents for single query"""
        return self.retriever.retrieve(question)
    
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
