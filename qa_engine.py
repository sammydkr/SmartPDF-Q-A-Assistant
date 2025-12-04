from typing import List, Dict
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import Config

class QAEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name=Config.OPENAI_MODEL,
            temperature=0.1
        )
        self.qa_chain = None
        
    def create_qa_chain(self, vector_store):
        """Create a QA chain with custom prompt"""
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in a clear and concise manner:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Create one first.")
        
        result = self.qa_chain({"query": question})
        
        return {
            "question": question,
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
    
    def ask_with_context(self, question: str, vector_store, k: int = 3) -> Dict:
        """Alternative method: search first, then answer"""
        # Search for relevant documents
        relevant_docs = vector_store.similarity_search(question, k=k)
        
        # Build context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt
        prompt = f"""
        Based on the following information, please answer the question.
        
        Information:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        # Get answer from LLM
        response = self.llm.predict(prompt)
        
        return {
            "question": question,
            "answer": response,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in relevant_docs
            ]
        }
