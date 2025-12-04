import os
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF into chunks for embedding"""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(raw_text)
        
        # Create LangChain documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} chunks from {pdf_path}")
        return documents
    
    def process_pdf_file(self, pdf_file) -> List[Document]:
        """Process PDF from uploaded file object"""
        # Save uploaded file temporarily
        temp_path = f"temp_{pdf_file.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        try:
            documents = self.process_pdf(temp_path)
            return documents
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
