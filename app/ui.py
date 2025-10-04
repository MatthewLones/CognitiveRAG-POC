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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .config-badge {
        background-color: #e1f5fe;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .groundedness-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    .grounded { background-color: #c8e6c9; color: #2e7d32; }
    .partially-grounded { background-color: #fff3e0; color: #f57c00; }
    .ungrounded { background-color: #ffcdd2; color: #d32f2f; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Cognitive RAG POC</div>', unsafe_allow_html=True)
st.markdown("**QMIND AI Club** - Demonstrating RAG evolution from Naïve to Enhanced techniques")

# Sidebar configuration
st.sidebar.header("Configuration")

# Config selector
config_options = {
    "naive": "Naïve RAG (Baseline)",
    "naive_plus": "Enhanced RAG (All Improvements)"
}

selected_config = st.sidebar.selectbox(
    "RAG Configuration",
    options=list(config_options.keys()),
    format_func=lambda x: config_options[x],
    index=1
)

# Display config badge
st.sidebar.markdown(f"""
<div class="config-badge">
    <strong>Active Config:</strong><br>
    {config_options[selected_config]}
</div>
""", unsafe_allow_html=True)

# Configuration details
with st.sidebar.expander("Configuration Details"):
    if selected_config == "naive":
        st.markdown("""
        **Baseline Features:**
        - Dense retrieval only
        - Simple chunking (400 tokens)
        - Basic citations
        - No self-check
        """)
    else:
        st.markdown("""
        **Enhanced Features:**
        - Hybrid retrieval (BM25 + Dense)
        - LLM re-ranking
        - Prompt fanning
        - Answerability gates
        - Self-check verification
        """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Query Interface")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="Ask a question about the AWS Well-Architected Framework...",
        height=100
    )
    
    # Query parameters
    col1a, col1b = st.columns(2)
    with col1a:
        max_results = st.slider("Max Results", 1, 10, 5)
    with col1b:
        show_debug = st.checkbox("Show Debug Info", value=False)
    
    # Submit button
    if st.button("🔍 Query Documents", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Processing query..."):
                try:
                    # TODO: Replace with actual API call
                    response = {
                        "answer": "This is a placeholder response. The RAG pipeline will be implemented in the next steps.",
                        "citations": [],
                        "confidence": 0.0,
                        "groundedness": "ungrounded",
                        "metadata": {"config": selected_config, "query": query}
                    }
                    
                    # Display results
                    st.subheader("Answer")
                    st.write(response["answer"])
                    
                    # Groundedness badge
                    groundedness_class = response["groundedness"].replace("_", "-")
                    st.markdown(f"""
                    <div class="groundedness-badge {groundedness_class}">
                        {response["groundedness"].title()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Citations
                    if response["citations"]:
                        with st.expander("Citations"):
                            for i, citation in enumerate(response["citations"]):
                                st.markdown(f"**{i+1}.** {citation.get('text', '')[:200]}...")
                    
                    # Debug info
                    if show_debug:
                        with st.expander("Debug Information"):
                            st.json(response["metadata"])
                            
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question.")

with col2:
    st.header("Demo Script")
    
    st.markdown("""
    **Quick Demo Steps:**
    
    1. **Easy Query**: "What is the Well-Architected Framework?"
       - Shows hybrid vs dense retrieval
    
    2. **Multi-facet**: "How do I improve security and performance?"
       - Demonstrates prompt fanning
    
    3. **Toggle Config**: Switch between Naïve and Enhanced
       - Compare groundedness badges
    
    4. **Performance**: Check response times and quality
    """)
    
    # Sample queries
    st.subheader("Sample Queries")
    sample_queries = [
        "What are the five pillars of the Well-Architected Framework?",
        "How do I implement security best practices?",
        "What are the cost optimization strategies?",
        "How do I ensure reliability in my architecture?",
        "What performance optimization techniques are recommended?"
    ]
    
    for i, sample_query in enumerate(sample_queries):
        if st.button(f"📝 {sample_query[:50]}...", key=f"sample_{i}", use_container_width=True):
            st.session_state.query_text = sample_query
            st.rerun()
    
    # Set query from sample
    if hasattr(st.session_state, 'query_text'):
        st.text_area("Selected Query:", value=st.session_state.query_text, disabled=True)

# Footer
st.markdown("---")
st.markdown("**Cognitive RAG POC** - Built for QMIND AI Club presentation")
st.markdown("Demonstrating measurable improvements from Naïve RAG to Enhanced techniques")
