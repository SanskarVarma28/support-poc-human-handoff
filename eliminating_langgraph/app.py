import openai
import os
import re
import numpy as np
from dotenv import load_dotenv
from prompt_faq import faq_text
from chat_history import ChatHistoryManager
from logger_config import logger
import traceback

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

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
        # logger.info(f"""Successfully created embeddings:{vectors}""")  # INFO log
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 2) -> list[dict]:
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
            logger.info("Found %d matches with top similarity score: %.4f.", 
                    len(results), results[0]["similarity"])  # INFO log
            return results
        except Exception as e:
            logger.error("Error during query: %s", str(e))  # Keep ERROR log for exceptions
            raise

def lookup_knowledge_base(query: str, retriever) -> str:
    """Consult the knowledge base to answer customer queries."""
    docs = retriever.query(query, k=2)
    logger.info(f"""Semantically matched docs:{docs}""")
    return "\n\n".join([doc["page_content"] for doc in docs])

def faq_llm_call(context: str, query: str, chat_history: str) -> str:
    try:
        if not context.strip():
            logger.info("Empty context received for query")  # INFO log
            return "I apologize, but I can only answer questions related to the FAQ content."
        
        system_prompt = """You are an FAQ assistant. Follow these rules strictly:
        1. Only answer based on the provided context and chat history
        2. If the question cannot be fully answered using the context, say you don't have enough information
        3. Keep answers brief and to the point
        4. Never make up information or use external knowledge
        5. Consider the conversation history when providing answers
        6. If the question is completely unrelated to the context, state that you can only answer questions related to the FAQ content
        7. Act as you are an AI assistant and gives the knowledge you have you dont need to mention that you dont have something in your
        knowledge base or so,if you are not able to continue giving the answer you can suggest them to talk to their sales representative for human assistance."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nChat History:\n{chat_history}\n\nCurrent Query: {query}"}
            ],
            max_tokens=250,
            temperature=0.2
        )
        
        logger.info(f"Received response from OpenAI: {response.choices[0].message.content}")  # INFO log
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error("Error in LLM call: %s\n%s", str(e), traceback.format_exc())  # Keep ERROR log for exceptions
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