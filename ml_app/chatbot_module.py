# -*- coding: utf-8 -*-
"""
RAG Chatbot Module for ML Django Application
Provides intelligent question-answering based on job dataset
"""
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai


class RAGChatbot:
    """RAG-based chatbot for job-related queries"""
    
    def __init__(self, csv_path=None, api_key=None):
        """
        Initialize the RAG chatbot
        
        Args:
            csv_path: Path to the AI jobs dataset CSV
            api_key: Google Gemini API key
        """
        self.model = None
        self.index = None
        self.chunks = None
        self.initialized = False
        
        # Set default CSV path if not provided
        if csv_path is None:
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'ai_job_dataset.csv'
            )
        
        self.csv_path = csv_path
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
    
    def initialize(self):
        """Initialize the RAG system: load data, create embeddings, build index"""
        if self.initialized:
            return
        
        try:
            # Load dataset
            df = pd.read_csv(self.csv_path)
            
            # Create documents from dataset
            documents = []
            metadatas = []
            
            for i, row in df.iterrows():
                doc_text = (
                    f"Job {row.get('job_id', 'N/A')}: {row.get('job_title', 'N/A')} at {row.get('company_name', 'N/A')}. "
                    f"Location: {row.get('company_location', 'N/A')}, employee residence: {row.get('employee_residence', 'N/A')}. "
                    f"Salary: {row.get('salary_usd', 'N/A')} {row.get('salary_currency', 'USD')}. "
                    f"Experience level: {row.get('experience_level', 'N/A')}, years of experience: {row.get('years_experience', 'N/A')}. "
                    f"Employment type: {row.get('employment_type', 'N/A')}, company size: {row.get('company_size', 'N/A')}. "
                    f"Remote ratio: {row.get('remote_ratio', 'N/A')}%. "
                    f"Industry: {row.get('industry', 'N/A')}. "
                    f"Required skills: {row.get('required_skills', 'N/A')}. "
                    f"Education: {row.get('education_required', 'N/A')}. "
                    f"Benefits score: {row.get('benefits_score', 'N/A')}."
                )
                
                documents.append(doc_text)
                metadatas.append({
                    "job_id": row.get("job_id", f"job_{i}"),
                    "job_title": row.get("job_title", "N/A"),
                    "company_location": row.get("company_location", "N/A"),
                    "salary_usd": row.get("salary_usd", "N/A"),
                    "company_name": row.get("company_name", "N/A"),
                    "industry": row.get("industry", "N/A")
                })
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            
            self.chunks = text_splitter.create_documents(documents, metadatas=metadatas)
            
            # Create embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            chunk_texts = [chunk.page_content for chunk in self.chunks]
            chunk_embeddings = self.model.encode(chunk_texts)
            
            # Create FAISS index
            d = chunk_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(np.array(chunk_embeddings).astype('float32'))
            
            self.initialized = True
            
        except Exception as e:
            print(f"Error initializing chatbot: {str(e)}")
            raise
    
    def answer_question(self, query: str, k: int = 3) -> dict:
        """
        Answer a question using RAG
        
        Args:
            query: User's question
            k: Number of relevant chunks to retrieve (default: 3 for speed)
            
        Returns:
            dict with 'answer' and 'context' keys
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Encode query
            query_embedding = self.model.encode([query]).astype('float32')
            
            # Search for similar chunks
            distances, indices = self.index.search(query_embedding, k)
            retrieved_chunks = [self.chunks[i].page_content for i in indices[0]]
            context = "\n\n".join(retrieved_chunks)
            
            # Generate answer with Gemini if available
            if self.gemini_model:
                prompt_template = f"""
Context:
{context}

Question: {query}

Answer in 2-3 sentences maximum. Be direct and concise.
"""
                response = self.gemini_model.generate_content(prompt_template)
                answer = response.text.strip()
            else:
                # Fallback: return context if no API available
                answer = f"Based on the available data:\n\n{context[:500]}..."
            
            return {
                'answer': answer,
                'context': context,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"Sorry, I encountered an error: {str(e)}",
                'context': '',
                'success': False
            }
    
    def get_quick_stats(self) -> dict:
        """Get quick statistics about the dataset"""
        if not self.initialized:
            self.initialize()
        
        return {
            'total_chunks': len(self.chunks) if self.chunks else 0,
            'index_size': self.index.ntotal if self.index else 0,
            'model_name': 'all-MiniLM-L6-v2'
        }


# Global instance
_chatbot_instance = None

def get_chatbot_instance(api_key=None):
    """Get or create the global chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot(api_key=api_key)
    return _chatbot_instance
