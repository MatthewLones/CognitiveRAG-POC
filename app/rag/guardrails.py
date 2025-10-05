"""
Guardrails and safety measures for Cognitive RAG POC
"""
from typing import List, Dict, Any, Tuple
import yaml
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GuardrailManager:
    """Manages answerability checks, self-verification, and safety measures"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        
        self.answerability_threshold = self.config.get('guardrails', {}).get('answerability_threshold', 0.7)
        self.self_check_enabled = self.config.get('guardrails', {}).get('self_check_enabled', False)
        self.max_citations = self.config.get('guardrails', {}).get('max_citations', 5)
        
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
    
    def check_answerability(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if the question can be answered with the retrieved chunks
        """
        try:
            import openai
            
            # Format context
            context = "\n\n".join([
                f"Document {i+1}:\n{chunk['content']}\nSource: {chunk['metadata']['source']}"
                for i, chunk in enumerate(chunks)
            ])
            
            # Create prompt
            prompt = f"""You are an expert evaluator. Determine if the provided context contains sufficient information to answer the question.

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
            
            # Call LLM
            response = openai.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback parsing
                result = {
                    "answerable": "true" in result_text.lower(),
                    "confidence": 0.5,
                    "reasoning": "Could not parse LLM response",
                    "available_info": "Unknown",
                    "missing_info": "Unknown"
                }
            
            # Ensure confidence is a float
            if isinstance(result.get('confidence'), str):
                try:
                    result['confidence'] = float(result['confidence'])
                except ValueError:
                    result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            print(f"Error in answerability check: {e}")
            return {
                "answerable": True,  # Default to answerable on error
                "confidence": 0.5,
                "reasoning": f"Error in answerability check: {str(e)}",
                "available_info": "Unknown",
                "missing_info": "Unknown"
            }
    
    def self_check(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify if the answer is grounded in the provided context
        """
        try:
            import openai
            
            # Format context
            context = "\n\n".join([
                f"Document {i+1}:\n{chunk['content']}\nSource: {chunk['metadata']['source']}"
                for i, chunk in enumerate(chunks)
            ])
            
            # Create prompt
            prompt = f"""You are a fact-checker. Verify if the answer is grounded in the provided context.

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
            
            # Call LLM
            response = openai.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback parsing
                result = {
                    "groundedness": "partially_grounded",
                    "supported_claims": ["Could not parse LLM response"],
                    "unsupported_claims": [],
                    "confidence": 0.5,
                    "corrections": "Could not parse LLM response"
                }
            
            # Ensure confidence is a float
            if isinstance(result.get('confidence'), str):
                try:
                    result['confidence'] = float(result['confidence'])
                except ValueError:
                    result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            print(f"Error in self-check: {e}")
            return {
                "groundedness": "partially_grounded",
                "supported_claims": [f"Error in self-check: {str(e)}"],
                "unsupported_claims": [],
                "confidence": 0.5,
                "corrections": f"Error in self-check: {str(e)}"
            }
    
    def filter_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and limit citations based on configuration
        """
        # Sort by score if available
        if citations and 'score' in citations[0]:
            citations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit to max citations
        return citations[:self.max_citations]
    
    def validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate response against safety criteria
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check if answer exists
        if not response.get('answer'):
            validation_result["valid"] = False
            validation_result["errors"].append("No answer provided")
        
        # Check confidence threshold
        confidence = response.get('confidence', 0)
        if confidence < self.answerability_threshold:
            validation_result["warnings"].append(f"Low confidence: {confidence}")
        
        # Check groundedness
        groundedness = response.get('groundedness', 'unknown')
        if groundedness == 'ungrounded':
            validation_result["warnings"].append("Answer is ungrounded")
        
        # Check citations
        citations = response.get('citations', [])
        if not citations:
            validation_result["warnings"].append("No citations provided")
        
        return validation_result
    
    def suggest_improvements(self, question: str, response: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements for the response
        """
        suggestions = []
        
        # Check answer length
        answer_length = len(response.get('answer', ''))
        if answer_length < 50:
            suggestions.append("Consider providing a more detailed answer")
        elif answer_length > 1000:
            suggestions.append("Consider making the answer more concise")
        
        # Check citations
        citations = response.get('citations', [])
        if len(citations) < 2:
            suggestions.append("Consider retrieving more supporting documents")
        
        # Check confidence
        confidence = response.get('confidence', 0)
        if confidence < 0.8:
            suggestions.append("Consider refining the query or expanding the document corpus")
        
        # Check groundedness
        groundedness = response.get('groundedness', 'unknown')
        if groundedness == 'partially_grounded':
            suggestions.append("Consider adding more specific citations")
        elif groundedness == 'ungrounded':
            suggestions.append("Consider reformulating the query or checking document relevance")
        
        return suggestions
    
    def get_safety_metrics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate safety metrics for the response
        """
        metrics = {
            "confidence": response.get('confidence', 0),
            "groundedness": response.get('groundedness', 'unknown'),
            "citation_count": len(response.get('citations', [])),
            "answer_length": len(response.get('answer', '')),
            "has_uncertainty": 'unsure' in response.get('answer', '').lower(),
            "has_citations": len(response.get('citations', [])) > 0
        }
        
        # Calculate overall safety score
        safety_score = 0.0
        
        # Confidence contributes 40%
        safety_score += metrics["confidence"] * 0.4
        
        # Groundedness contributes 30%
        groundedness_scores = {
            'grounded': 1.0,
            'partially_grounded': 0.6,
            'ungrounded': 0.2
        }
        safety_score += groundedness_scores.get(metrics["groundedness"], 0.2) * 0.3
        
        # Citations contribute 20%
        citation_score = min(metrics["citation_count"] / 5, 1.0)
        safety_score += citation_score * 0.2
        
        # Answer quality contributes 10%
        if metrics["has_citations"] and not metrics["has_uncertainty"]:
            safety_score += 0.1
        
        metrics["safety_score"] = min(safety_score, 1.0)
        
        return metrics

# Example usage
if __name__ == "__main__":
    guardrail_manager = GuardrailManager()
    
    # Example chunks
    example_chunks = [
        {
            "content": "The Well-Architected Framework provides guidance to help you build secure, high-performing, resilient, and efficient infrastructure.",
            "metadata": {"source": "test.pdf", "chunk_id": "0"}
        }
    ]
    
    # Test answerability check
    answerability = guardrail_manager.check_answerability(
        "What is the Well-Architected Framework?",
        example_chunks
    )
    
    print("Answerability check:", answerability)
    
    # Test self-check
    self_check = guardrail_manager.self_check(
        "The Well-Architected Framework is a set of best practices for building cloud applications.",
        example_chunks
    )
    
    print("Self-check:", self_check)
