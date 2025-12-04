import streamlit as st
import os
from pdf_processor import PDFProcessor
from vector_store import VectorStoreManager
from qa_engine import QAEngine
from config import Config

# Initialize components
processor = PDFProcessor()
vector_manager = VectorStoreManager()
qa_engine = QAEngine()

def main():
    st.set_page_config(page_title="SmartPDF Q&A Assistant", page_icon="📚")
    
    st.title("📚 SmartPDF Q&A Assistant")
    st.markdown("Upload PDFs and ask questions about their content using AI!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Upload a PDF file")
        st.markdown("2. Click 'Process Document'")
        st.markdown("3. Ask questions about the content")
    
    # Main content area
    tab1, tab2 = st.tabs(["Upload & Process", "Ask Questions"])
    
    with tab1:
        st.header("Upload PDF Document")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        documents = processor.process_pdf_file(uploaded_file)
                        
                        # Create vector store
                        vector_store = vector_manager.create_vector_store(
                            documents, 
                            store_name=uploaded_file.name.replace(".pdf", "")
                        )
                        
                        # Create QA chain
                        qa_engine.create_qa_chain(vector_store)
                        
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_file = uploaded_file.name
                        st.session_state.is_processed = True
                        
                        st.success("✅ Document processed successfully!")
                        st.info(f"Created {len(documents)} text chunks")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
    
    with tab2:
        st.header("Ask Questions")
        
        if not st.session_state.get("is_processed", False):
            st.warning("Please upload and process a PDF document first.")
            return
        
        st.info(f"Currently using: {st.session_state.get('processed_file', 'No file')}")
        
        # Question input
        question = st.text_input(
            "Enter your question about the document:",
            placeholder="e.g., What are the main findings of this research?"
        )
        
        if question:
            with st.spinner("Searching for answers..."):
                try:
                    # Get answer
                    result = qa_engine.ask_question(question)
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.write(result["answer"])
                    
                    # Display sources
                    with st.expander("View Source Context"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.write(source["content"])
                            st.markdown(f"*From: {source['metadata'].get('source', 'Unknown')}*")
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **About this project:** 
        This tool uses RAG (Retrieval-Augmented Generation) with OpenAI embeddings 
        and FAISS vector search to provide accurate answers from PDF documents.
        """
    )

if __name__ == "__main__":
    main()
