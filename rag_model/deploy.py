from qdrant_client import QdrantClient
from featurization_pipeline import featurize_text

QDRANT_URI = ""
QDRANT_API_KEY = ""
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
