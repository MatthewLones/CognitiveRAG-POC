"""
Streamlit UI for Cognitive RAG POC
"""
import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Cognitive RAG POC",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c5aa0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .mode-toggle {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .prompt-section {
        background-color: #f1f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        border-radius: 0.5rem !important;
        border: 2px solid #e9ecef !important;
        transition: border-color 0.3s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
    }
    .stButton button {
        border-radius: 0.5rem !important;
        background-color: #4a90e2 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background-color: #357abd !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3) !important;
    }
    .stRadio > div {
        gap: 1rem;
    }
    .stRadio > div > label {
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stRadio > div > label:hover {
        background-color: #f1f8ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Cognitive RAG POC</div>', unsafe_allow_html=True)

# Mode toggle
st.markdown('<div class="mode-toggle">', unsafe_allow_html=True)
selected_config = st.radio(
    "AI Mode:",
    ["naive", "naive_plus"],
    format_func=lambda x: "Naive RAG" if x == "naive" else "Enhanced RAG",
    horizontal=True,
    index=1
)
st.markdown('</div>', unsafe_allow_html=True)

# Query input
query = st.text_input(
    "Ask a question about the AWS Well-Architected Framework:",
    placeholder="What are the five pillars of the Well-Architected Framework?",
    key="query_input"
)

# Submit button
if st.button("🔍 Query", type="primary", use_container_width=True):
    if query.strip():
        with st.spinner("Processing query..."):
            try:
                # Make API call to FastAPI backend
                api_url = "http://localhost:8000"
                
                # Check if API is running
                try:
                    health_response = requests.get(f"{api_url}/health", timeout=5)
                    if health_response.status_code != 200:
                        st.error("FastAPI server is not running. Please start it with: `uvicorn app.api:app --reload`")
                        st.stop()
                except requests.exceptions.RequestException:
                    st.error("Cannot connect to FastAPI server. Please start it with: `uvicorn app.api:app --reload`")
                    st.stop()
                
                # Make the query request
                query_data = {
                    "query": query,
                    "config": selected_config,
                    "max_results": 5
                }
                
                response = requests.post(
                    f"{api_url}/query",
                    json=query_data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("**Answer:**")
                    st.markdown(result["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display processing data
                    if "processing_metrics" in result.get("metadata", {}):
                        metrics = result["metadata"]["processing_metrics"]
                        
                        # Show actual prompts and processing steps
                        st.markdown("**🔧 Processing Data:**")
                        
                        # Retrieval method and chunks
                        st.markdown(f"""
                        <div class="prompt-section">
                            <strong>Retrieval:</strong> {metrics['retrieval_metrics']['retrieval_method'].upper()}<br>
                            <strong>Chunks Retrieved:</strong> {metrics['retrieval_metrics']['chunks_retrieved']}<br>
                            <strong>Reranking Applied:</strong> {'Yes' if metrics['retrieval_metrics']['reranking_applied'] else 'No'}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prompt fanning
                        if metrics['prompt_fanning']['enabled']:
                            st.markdown(f"""
                            <div class="prompt-section">
                                <strong>Prompt Fanning:</strong> Enabled<br>
                                <strong>Sub-queries Generated:</strong> {metrics['prompt_fanning']['subqueries_generated']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show generated subqueries
                            if metrics['prompt_fanning']['generated_subqueries']:
                                st.markdown("**Generated Sub-queries:**")
                                for i, subquery in enumerate(metrics['prompt_fanning']['generated_subqueries'], 1):
                                    st.markdown(f"{i}. {subquery}")
                            
                            # Show decomposition prompt
                            if metrics['prompt_fanning']['decomposition_prompt']:
                                with st.expander("Decomposition Prompt"):
                                    st.text(metrics['prompt_fanning']['decomposition_prompt'])
                        
                        # Guardrails
                        st.markdown(f"""
                        <div class="prompt-section">
                            <strong>Answerability Check:</strong> {'Enabled' if metrics['guardrails']['answerability_enabled'] else 'Disabled'}<br>
                            <strong>Self-Check:</strong> {'Enabled' if metrics['guardrails']['self_check_enabled'] else 'Disabled'}<br>
                            <strong>Dynamic Citations:</strong> {'Enabled' if metrics['guardrails']['dynamic_citations'] else 'Disabled'}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Chunk scores
                        if metrics['chunk_scores']:
                            st.markdown("**📊 Chunk Scores:**")
                            for chunk in metrics['chunk_scores'][:3]:  # Show top 3
                                st.markdown(f"- {chunk['chunk_id']}: {chunk['retrieval_score']:.3f} ({chunk['source']})")
                        
                        # Show actual prompts used
                        if metrics.get('prompts_used', {}).get('answer_prompt'):
                            with st.expander("Answer Generation Prompt"):
                                st.text(metrics['prompts_used']['answer_prompt'])
                        
                        # Raw JSON data
                        with st.expander("Raw Processing Data"):
                            st.json(metrics)
                    
                    # Citations
                    if result["citations"]:
                        st.markdown(f"**📚 Citations ({len(result['citations'])}):**")
                        for i, citation in enumerate(result["citations"][:3]):  # Show top 3
                            st.markdown(f"{i+1}. {citation.get('text', '')[:100]}...")
                            st.markdown(f"   *Source: {citation.get('source', 'Unknown')}*")
                
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                st.error("Query timed out. The server might be processing a large document.")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.markdown("**Cognitive RAG POC** - Demonstrating RAG evolution from Naïve to Enhanced techniques")