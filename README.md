# RAG (Retrieval-Augmented Generation) Project

This project implements a Retrieval-Augmented Generation (RAG) system using two types of large language models (LLMs) - OpenAI's GPT and Gemini. The system enables querying information from an external database and generating responses based on the context retrieved.

## Overview

RAG is a hybrid system combining the strengths of retrieval-based models and generative models to provide accurate and context-rich responses. Here's a breakdown of how this project works:

1. **Query Processing**: User inputs are passed through a retrieval system to identify relevant documents or data stored in a MongoDB database.
2. **Embedding Model**: The query is transformed into vector embeddings using the selected `EMBEDDING_MODEL`, which ensures accurate matching in the retrieval phase.
3. **Retrieval Phase**: The embeddings are used to search for relevant information in the database collection (`DB_COLLECTION`), which holds documents or articles that can augment the generation process.
4. **LLM Generation**: After the relevant documents are retrieved, they are passed into the chosen LLM (OpenAI GPT or Gemini) to generate a comprehensive and contextually relevant response.
5. **Result Delivery**: The final output is a combination of the retrieved data and the generative response, delivering enhanced and informative responses to the user.

### Configuration

You will need to create a `.env` file in the root directory to store API keys and other configuration settings. Below is an example of the required fields:

```bash
MONGODB_URI=your_mongodb_connection_string
DB_NAME=your_database_name
DB_COLLECTION=your_collection_name
EMBEDDING_MODEL=your_embedding_model_choice
OPENAI_API_KEY=your_openai_api_key
GEMINI_KEY=your_gemini_api_key
