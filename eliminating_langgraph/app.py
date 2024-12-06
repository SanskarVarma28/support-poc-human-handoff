import openai
import os
import re
import numpy as np
from dotenv import load_dotenv
from prompt_faq import faq_text
from chat_history import ChatHistoryManager
from logger_config import setup_logger
import traceback

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
# Set up your OpenAI API key
# Set up loggers
main_logger = setup_logger('main')
retriever_logger = setup_logger('retriever')
llm_logger = setup_logger('llm')

# Initialize the history manager
history_manager = ChatHistoryManager()

class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        retriever_logger.info("Successfully created embeddings")
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 2) -> list[dict]:
        retriever_logger.debug("Processing query: %s", query[:50] + "..." if len(query) > 50 else query)
        try:
            embed = self._client.embeddings.create(
                model="text-embedding-3-small", 
                input=[query]
            )
            scores = np.array(embed.data[0].embedding) @ self._arr.T
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
            results = [
                {**self._docs[idx], "similarity": scores[idx]} 
                for idx in top_k_idx_sorted
            ]
            retriever_logger.debug("Found %d matches with top similarity score: %.4f", 
                                 len(results), results[0]["similarity"])
            return results
        except Exception as e:
            retriever_logger.error("Error during query: %s", str(e))
            raise

def lookup_knowledge_base(query: str, retriever) -> str:
    """Consult the knowledge base to answer customer queries."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

def faq_llm_call(context: str, query: str, chat_history: str) -> str:
    llm_logger.debug("Processing LLM call - Query: %s", 
                    query[:50] + "..." if len(query) > 50 else query)
    try:
        if not context.strip():
            llm_logger.warning("Empty context received for query")
            return "I apologize, but I can only answer questions related to the FAQ content."
        
        system_prompt = """You are an FAQ assistant. Follow these rules strictly:
        1. Only answer based on the provided context and chat history
        2. If the question cannot be fully answered using the context, say you don't have enough information
        3. Keep answers brief and to the point
        4. Never make up information or use external knowledge
        5. Consider the conversation history when providing answers
        6. If the question is completely unrelated to the context, state that you can only answer questions related to the FAQ content"""
        
        llm_logger.debug("Sending request to OpenAI with context length: %d, history length: %d", 
                        len(context), len(chat_history))
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nChat History:\n{chat_history}\n\nCurrent Query: {query}"}
            ],
            max_tokens=250,
            temperature=0.2
        )
        
        llm_logger.debug("Received response from OpenAI, length: %d", 
                        len(response.choices[0].message.content))
        return response.choices[0].message.content
        
    except Exception as e:
        llm_logger.error("Error in LLM call: %s\n%s", str(e), traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Initialize the vector store retriever
docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]
retriever = VectorStoreRetriever.from_docs(docs, client)

# Example usage
# if __name__ == "__main__":
#     # Example FAQ prompt
#     faq_prompt = faq_text
#     # User query
#     user_query = "hubspot vs supersales ??"
    
#     # Get response
#     response = faq_llm_call(faq_prompt, user_query)
#     print("AI Response:", response)
