import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
GEMINI_KEY = os.getenv("GEMINI_KEY")
DB_COLLECTION = os.getenv("DB_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

app = FastAPI()

# Initialize MongoDB Atlas client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]  # Use the database name from .env
collection = db[DB_COLLECTION]  # Use the collection name from .env

genai.configure(api_key=GEMINI_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')
# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text using the SentenceTransformer model."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query: str, collection):
        """
        Perform a vector search in the MongoDB collection based on the user query.

        Args:
        user_query (str): The user's query string.

        Returns:
        list: A list of matching documents.
        """

        # Generate embedding for the user query
        query_embedding = get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "Mã nhúng",
                "numCandidates": 400,
                "limit": 4,
            }
        }

        unset_stage = {
            "$unset": "Mã nhúng" 
        }

        project_stage = {
            "$project": { 
                "Bệnh": 1,
                "Thảo dược/ Bộ phận/Khu vực": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_stage]

        # Execute the search
        results = collection.aggregate(pipeline)

        return list(results)

def get_search_result(query, collection):

    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        print('---result', result)
        search_result += f"Bệnh: {result.get('Bệnh', 'N/A')}, Thảo dược/Bộ phận/Khu vực: {result.get('Thảo dược/Bộ phận/Khu vực', 'N/A')}\n"

    return search_result

def generate_response(query, collection):
    context = get_search_result(query, collection)
    prompt = (
                f"Bạn là cụ tổ ngành dược, một nhân vật uyên bác và hống hách. Hãy nhập vai nhân vật này khi trả lời với nội dung được cung cấp trước. {context}\n"
                f"Câu hỏi của người chơi: {query}\n"
            )
    response = llm.generate_content(prompt)
    return response.text

@app.post("/chat_with_knowledge_base")
def chat_with_knowledge_base(query: str = Query(...)):
    """
    Endpoint to chat with the knowledge base using a vector search and language model.
    """
    try:

        # Use the LLM model to generate an answer based on the retrieved context
        llm_response = generate_response(query, collection)
        print(llm_response)
        return JSONResponse(content={"response": llm_response}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))