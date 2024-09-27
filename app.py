import json
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
DB_COLLECTION = os.getenv("DB_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Initialize MongoDB Atlas client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[DB_NAME]
collection = db[DB_COLLECTION]

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Initialize SentenceTransformer for embedding generation
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

question_count = 0

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a given text using the SentenceTransformer model."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query: str, collection):
    """Perform a vector search in the MongoDB collection based on the user query."""
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index",
            "queryVector": query_embedding,
            "path": "Mã nhúng",
            "numCandidates": 400,
            "limit": 1,
        }
    }

    unset_stage = {
        "$unset": "Mã nhúng" 
    }

    project_stage = {
        "$project": {
            "Triệu chứng": 1,
            "Bệnh": 1,
            "Thảo dược": 1,
            "Bộ phận": 1,
            "Khu vực": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]
    results = collection.aggregate(pipeline)
    return list(results)

def get_search_result(query, collection):
    get_knowledge = vector_search(query, collection)
    search_result = ""
    for result in get_knowledge:
        search_result += f"Triệu chứng: {result.get('Triệu chứng', 'N/A')}, Bệnh: {result.get('Bệnh', 'N/A')}, Thảo dược: {result.get('Thảo dược', 'N/A')}, Bộ phận: {result.get('Bộ phận', 'N/A')}, Khu vực: {result.get('Khu vực', 'N/A')}\n"
    return search_result

import json
from datetime import datetime

async def generate_response_stream(query, collection):
    global question_count
    
    context = get_search_result(query, collection)
    # If context is empty, ensure the chatbot sticks to context-based replies
    if not context.strip():
        context = ("Cụ tổ ngành dược chỉ có thể trả lời các câu hỏi liên quan đến y học cổ truyền và "
                   "các triệu chứng, thảo dược trong cơ sở dữ liệu của trò chơi. Hãy hỏi về "
                   "triệu chứng hoặc bệnh mà con đang gặp phải, cụ sẽ cố gắng giúp đỡ.")
        
    if question_count < 3:
        system_prompt = f"""
        Bạn là cụ tổ ngành dược, một nhân vật uyên bác, và chỉ được trả lời các câu hỏi về y học cổ truyền, các triệu chứng, bệnh, thảo dược, bộ phận của thảo dược và khu vực để tìm thảo dược đó.  
        Triệu chứng và bệnh lý của người chơi liên quan đến: {context}.
        Nếu người chơi hỏi những câu không liên quan, hãy từ chối trả lời và khéo léo hướng dẫn họ quay lại chủ đề chính. 
        Xưng hô với người chơi là "con" để tạo cảm giác gần gũi.
        Hãy trả lời với tư cách là cụ tổ ngành dược và không đề cập đến các yếu tố bên ngoài như đi khám bác sĩ hiện đại hay là điều trị tại cơ sở bệnh viện nào khác.
        """
    else:
        system_prompt = """
        Con đã hỏi nhiều câu rồi, cụ sẽ không trả lời trực tiếp nữa mà sẽ đưa ra gợi ý bằng các câu đố.
        Con hãy thử suy nghĩ và giải đáp những câu đố này để tìm ra bệnh và thảo dược mà con cần.
        """
    
    user_prompt = f"Câu hỏi của người chơi: {query}\n\nCụ tổ ngành dược trả lời:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    question_count += 1

    print(f"Query received: {query}")
    print(f"Context: {context}")

    async for chunk in await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    ):
        try:
            delta = chunk.choices[0].delta
            response_id = chunk.id
            timestamp = int(datetime.utcnow().timestamp())

            if delta.content:
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": timestamp,
                    "model": "gpt-3.5-turbo",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": delta.content,
                                "refusal": None
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            print(f"Error in OpenAI stream: {e}")
            error_data = {
                "id": "unknown",
                "object": "chat.completion.chunk",
                "created": int(datetime.utcnow().timestamp()),
                "model": "gpt-3.5-turbo",
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "Oops!! Something wrong! Please contact thangchiba@gmail.com",
                            "refusal": None
                        },
                        "logprobs": None,
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            break

    done_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "gpt-3.5-turbo",
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat_with_knowledge_base")
async def chat_with_knowledge_base(query: str = Form(...)):
    """Endpoint to chat with the knowledge base using a vector search and OpenAI's language model."""
    try:
        return StreamingResponse(
            content=generate_response_stream(query, collection),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
