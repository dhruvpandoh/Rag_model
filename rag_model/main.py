from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, ScoredPoint
from featurization_pipeline import featurize_text  
import logging

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer from HuggingFace with authentication token
HUGGINGFACE_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained("ArmaanDhande/rag_model_t5_AI", use_auth_token=HUGGINGFACE_TOKEN)
model = AutoModelForSeq2SeqLM.from_pretrained("ArmaanDhande/rag_model_t5_AI", use_auth_token=HUGGINGFACE_TOKEN)

# Initialize model and tokenizer for embeddings
EMBEDDING_MODEL_NAME = "bert-base-uncased"
embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

# Qdrant Configuration
QDRANT_HOST = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "rzwHZa71bmoNZzJ2YlEvuwWoH8-2WifxSLywPqZc-o8zkaKilb3z1w"
qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
QDRANT_COLLECTION_NAME = "star_charts" 

class QuestionRequest(BaseModel):
    question: str

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/")
def root():
    return {"message": "Welcome to the ROS2 RAG system!"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        logging.info(f"Received question: {request.question}")
        
        # Enhance the question with specific context requirements
        enhanced_question = f"{request.question}"
        
        question_vector = featurize_text(
            enhanced_question,
            embedding_tokenizer,
            embedding_model,
            device
        )
        
        # Increase the number of retrieved contexts
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=question_vector.tolist(),
            limit=5  # Increased from 3 to get more context
        )
        
        # Debug log the search results
        logging.info(f"Number of search results: {len(search_results)}")
        for idx, result in enumerate(search_results):
            logging.info(f"Result {idx + 1} score: {result.score}")
            logging.info(f"Result {idx + 1} payload preview: {str(result.payload)[:100]}")

        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant context found")

        # Extract and format context more effectively
        relevant_contexts = [result.payload.get("text", "") for result in search_results]
        relevant_context = " ".join(filter(None, relevant_contexts))
        
        if not relevant_context.strip():
            raise HTTPException(status_code=404, detail="Retrieved context is empty")
            
        # Create a more structured prompt
        input_text = f"""
        Based on the following ROS2 navigation documentation, provide a detailed answer about navigating to a specific pose.
        Include information about:
        1. How to set and send pose goals
        2. How the navigation stack handles replanning
        3. Key components involved in the process
        4. Error handling and recovery behaviors

        Question: {request.question}
        
        Context: {relevant_context}
        
        Provide a clear, step-by-step response focusing on practical implementation.
        """
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,  # Increased for more context
            truncation=True
        )

        logging.info("Generating answer using the fine-tuned model...")
        outputs = model.generate(
            **inputs,
            max_length=300,      
            min_length=100,  
            num_beams=5,
            length_penalty=2.0,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not answer.strip():
            raise HTTPException(status_code=500, detail="Model generated empty response")

        logging.info(f"Generated answer: {answer}")
        
        return {
            "question": request.question,
            "answer": answer,
            "context_length": len(relevant_context)
        }
        
    except HTTPException as e:
        logging.error(f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")


