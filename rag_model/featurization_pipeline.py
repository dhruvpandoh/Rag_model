# app/featurization_pipeline.py

#!/usr/bin/env python
# coding: utf-8

import logging
import os
import uuid
from bs4 import BeautifulSoup

import numpy as np
import pymongo
import torch
from clearml import Task
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration values
MONGO_URI = ""
QDRANT_HOST = ""
QDRANT_PORT = 6333
QDRANT_API_KEY = ""
QDRANT_COLLECTION_NAME = "star_charts"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 500
MAX_TEXT_LENGTH = 1000

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ClearML Task
task = Task.init(project_name="ROS2 RAG System", task_name="Featurization Pipeline")


# Extracts and cleans text from HTML content.
def extract_text_from_html(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())
    except Exception as e:
        logging.error(f"Failed to extract text from HTML: {e}")
        return ""

# Truncates text to the specified maximum length without cutting off in the middle of a word.
def truncate_text(text, max_length=MAX_TEXT_LENGTH):
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + '...'

# Generates embeddings for the given text using the loaded model.
def featurize_text(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.squeeze()
    except Exception as e:
        logging.error(f"Failed to featurize text: {e}")
        return None

# Ensure Qdrant collection exists with the correct vector configuration.
def ensure_qdrant_collection(qdrant_client, collection_name, vector_size):
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name not in collection_names:
            logging.info(f"Creating Qdrant collection: {collection_name}")
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logging.info(f"Collection '{collection_name}' created successfully.")
        else:
            logging.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logging.error(f"Error ensuring Qdrant collection: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(Exception),
    reraise=True
)

# Upserts a batch of points into the specified Qdrant collection.
def upsert_points(qdrant_client, collection_name, points):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logging.info(f"Successfully upserted {len(points)} points to Qdrant.")
    except Exception as e:
        logging.error(f"Failed to upsert points: {e}")
        raise

def main():
    # Connect to MongoDB
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["ros2_rag"]
        collection = db["raw_data"]
        logging.info("Connected to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        task.close()
        return

    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            timeout=60
        )
        logging.info("Connected to Qdrant successfully.")
    except Exception as e:
        logging.error(f"Failed to connect to Qdrant: {e}")
        task.close()
        return

    # Load the tokenizer and model
    try:
        logging.info("Loading tokenizer and model for embeddings...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        embedding_dimension = model.config.hidden_size
        logging.info(f"Tokenizer and model loaded successfully. Embedding dimension: {embedding_dimension}.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer and model: {e}")
        task.close()
        return

    # Ensure Qdrant collection exists
    try:
        ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME, embedding_dimension)
    except Exception as e:
        logging.error(f"Failed to ensure Qdrant collection: {e}")
        task.close()
        return

    # Prepare data for insertion into Qdrant
    payloads = []
    vectors = []
    ids = []

    try:
        documents_cursor = collection.find()
        documents = list(documents_cursor)
        total_documents = len(documents)
        if total_documents == 0:
            logging.info("No documents found in MongoDB collection. Exiting.")
            task.close()
            return
        logging.info(f"Found {total_documents} documents in MongoDB collection.")
    except Exception as e:
        logging.error(f"Failed to retrieve documents from MongoDB: {e}")
        task.close()
        return

    logging.info("Processing documents and generating embeddings...")
    points = []
    for doc in tqdm(documents, desc="Processing Documents"):
        try:
            text = doc.get("text_content", "").strip()
            if not text:
                html_content = doc.get("content", "")
                text = extract_text_from_html(html_content)
                if not text:
                    logging.warning(f"Skipping document with ID {doc.get('_id')} due to missing text.")
                    continue

            truncated_text = truncate_text(text)
            vector = featurize_text(truncated_text, tokenizer, model, device)
            if vector is None or not isinstance(vector, np.ndarray) or np.isnan(vector).any():
                logging.error(f"Invalid vector generated for document ID {doc.get('_id')}. Skipping.")
                continue

            vector = vector.astype(float).tolist()
            payload = {
                "text": truncated_text,
                "source": doc.get("source", "unknown"),
                "url": doc.get("url", ""),
                "video_id": doc.get("video_id", "")
            }
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)

            # Upsert in batches
            if len(points) >= BATCH_SIZE:
                upsert_points(qdrant_client, QDRANT_COLLECTION_NAME, points)
                points = []

        except Exception as e:
            logging.error(f"Failed to process document ID {doc.get('_id')}: {e}")

    # Upsert any remaining points
    if points:
        upsert_points(qdrant_client, QDRANT_COLLECTION_NAME, points)

    logging.info(f"Successfully inserted {total_documents} documents into Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
    task.close()

if __name__ == "__main__":
    main()
