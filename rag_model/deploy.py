from qdrant_client import QdrantClient
from featurization_pipeline import featurize_text

QDRANT_URI = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "r7tCBpWwRoewwOOIP-VuZgRoka_YZGvwY24SzEK1K85CyU1-Ii772Q"
qdrant_client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)

def query_rag(question):
    # Convert question to vector using the same featurization as training
    question_vector = featurize_text(question) 
    # Retrieve relevant data from Qdrant
    search_results = qdrant_client.search(
        collection_name="star_charts",
        query_vector=question_vector.tolist(),
        limit=5
    )
    context = " ".join([result.payload.get("content", "") for result in search_results])
