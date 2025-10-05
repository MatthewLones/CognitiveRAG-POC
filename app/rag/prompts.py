"""
Prompt templates for Cognitive RAG POC
"""
from typing import Dict, Any
import yaml

class PromptManager:
    """Manages prompt templates for different RAG components"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        
        self.llm_model = self.config['models']['llm_model']
    
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
    
    def get_answer_prompt(self) -> str:
        """Get the main answer generation prompt"""
        return """You are an expert assistant helping users find information from documents. Use the provided context to answer the question accurately and comprehensively.

Question: {question}

Context from documents:
{context}

Instructions:
1. Answer the question based ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Include specific citations by referencing "Document 1", "Document 2", etc.
4. Structure your answer clearly with proper formatting:
   - Use bullet points for lists
   - Use **bold** for key terms
   - Use *italics* for emphasis
   - Use numbered lists for steps or sequences
5. Be concise but complete in your answer
6. If you're unsure about something, express that uncertainty
7. Format your response for easy reading with proper markdown

Answer:"""
    
    def get_rerank_prompt(self) -> str:
        """Get the re-ranking prompt"""
        return """You are a document retrieval expert. Given a query and a list of document chunks, rank them by relevance to the query.

Query: {query}

Document chunks:
{chunks}

Please rank these chunks by relevance to the query. Return only the numbers in order of relevance, separated by commas.

Example: 3,1,5,2,4"""
    
    def get_decomposition_prompt(self) -> str:
        """Get the query decomposition prompt"""
        return """Break down this question into {max_subqueries} specific sub-queries to improve document retrieval:

Original question: {question}

Sub-query types to consider: {subquery_types}

Return {max_subqueries} specific sub-queries, one per line:"""
    
    def get_answerability_prompt(self) -> str:
        """Get the answerability check prompt"""
        return """You are an expert evaluator. Determine if the provided context contains sufficient information to answer the question.

Question: {question}

Context:
{context}

Evaluate:
1. Can this question be answered based on the provided context?
2. Rate confidence from 0.0 (no information) to 1.0 (complete information)
3. If answerable, what specific information is available?
4. If not answerable, what information is missing?

Respond in JSON format:
{{
    "answerable": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "available_info": "what's available",
    "missing_info": "what's missing"
}}"""
    
    def get_self_check_prompt(self) -> str:
        """Get the self-check verification prompt"""
        return """You are a fact-checker. Verify if the answer is grounded in the provided context.

Question: {question}

Answer to verify: {answer}

Context used:
{context}

Instructions:
1. Check each claim in the answer against the context
2. Identify any claims that cannot be supported by the context
3. Determine the overall groundedness of the answer
4. Suggest corrections if needed

Respond in JSON format:
{{
    "groundedness": "grounded/partially_grounded/ungrounded",
    "supported_claims": ["list of supported claims"],
    "unsupported_claims": ["list of unsupported claims"],
    "confidence": 0.0-1.0,
    "corrections": "suggested corrections if needed"
}}"""
    
    def get_hybrid_retrieval_prompt(self) -> str:
        """Get the hybrid retrieval explanation prompt"""
        return """Explain why these documents were retrieved for the query:

Query: {query}

Retrieved documents:
{retrieved_docs}

Explain:
1. Why each document is relevant to the query
2. How the hybrid retrieval (BM25 + dense embeddings) helped find these documents
3. What makes these documents good matches for the query

Keep the explanation concise and technical."""
    
    def get_prompt_fanning_prompt(self) -> str:
        """Get the prompt fanning explanation prompt"""
        return """Explain how query decomposition helped retrieve better results:

Original query: {original_query}

Sub-queries generated:
{subqueries}

Documents retrieved for each sub-query:
{retrieval_results}

Explain:
1. How breaking down the query improved retrieval
2. Which sub-queries found the most relevant documents
3. How the results were merged and deduplicated

Keep the explanation concise and technical."""
    
    def get_guardrail_prompt(self) -> str:
        """Get the guardrail explanation prompt"""
        return """Explain the safety and trust measures applied to this response:

Query: {query}

Answer: {answer}

Groundedness: {groundedness}

Safety measures applied:
- Answerability check: {answerability_check}
- Self-check verification: {self_check}
- Confidence threshold: {confidence_threshold}

Explain:
1. How these measures ensure answer quality
2. What the groundedness level means
3. Why these measures are important for RAG systems

Keep the explanation concise and technical."""
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with provided variables"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Missing variable in prompt template: {e}")
            return template
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are an expert assistant for the Cognitive RAG POC system. You help users find information from documents using advanced retrieval techniques.

Your capabilities include:
- Hybrid retrieval (BM25 + dense embeddings)
- Query decomposition and parallel processing
- Answerability checking and self-verification
- Grounded response generation with citations

Always provide accurate, well-cited responses based on the provided context. If information is insufficient, clearly state limitations."""
    
    def get_evaluation_prompt(self) -> str:
        """Get the evaluation prompt for LLM-as-judge"""
        return """You are an expert evaluator for RAG systems. Evaluate the quality of this response:

Question: {question}

Answer: {answer}

Context used: {context}

Evaluate on these dimensions (0-5 scale):
1. Groundedness: How well is the answer supported by the context?
2. Relevance: How relevant is the answer to the question?
3. Completeness: How complete is the answer?
4. Clarity: How clear and well-structured is the answer?

Provide scores and brief justifications for each dimension.

Respond in JSON format:
{{
    "groundedness": {"score": 0-5, "justification": "explanation"},
    "relevance": {"score": 0-5, "justification": "explanation"},
    "completeness": {"score": 0-5, "justification": "explanation"},
    "clarity": {"score": 0-5, "justification": "explanation"},
    "overall_score": 0-5
}}"""

# Example usage
if __name__ == "__main__":
    prompt_manager = PromptManager()
    
    # Test prompt formatting
    answer_prompt = prompt_manager.get_answer_prompt()
    formatted = prompt_manager.format_prompt(
        answer_prompt,
        question="What is the Well-Architected Framework?",
        context="The Well-Architected Framework provides guidance..."
    )
    
    print("Formatted prompt:")
    print(formatted)
