from typing import List, Dict
from datetime import datetime
from logger_config import setup_logger

logger = setup_logger('chat_history')

class ChatHistoryManager:
    def __init__(self):
        self.histories: Dict[str, List[Dict]] = {}
        self.max_history_length = 10
        logger.info("Initialized ChatHistoryManager with max_history_length=%d", self.max_history_length)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a new message to the chat history."""
        logger.debug("Adding message for session %s: role=%s, content_length=%d", 
                    session_id, role, len(content))
        
        if session_id not in self.histories:
            logger.info("Creating new history for session %s", session_id)
            self.histories[session_id] = []
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.histories[session_id].append(message)
        
        if len(self.histories[session_id]) > self.max_history_length:
            logger.debug("Trimming history for session %s to max length", session_id)
            self.histories[session_id] = self.histories[session_id][-self.max_history_length:]

    def get_history(self, session_id: str) -> List[Dict]:
        """Get the chat history for a session."""
        history = self.histories.get(session_id, [])
        logger.debug("Retrieved history for session %s: %d messages", 
                    session_id, len(history))
        return history

    def format_history_for_prompt(self, session_id: str) -> str:
        """Format the chat history for the LLM prompt."""
        history = self.get_history(session_id)
        formatted_history = []
        
        for msg in history:
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            formatted_history.append(f"{role_prefix}{msg['content']}")
        
        logger.debug("Formatted history for session %s: %d messages", 
                    session_id, len(history))
        return "\n".join(formatted_history)

    def clear_history(self, session_id: str) -> None:
        """Clear the chat history for a session."""
        if session_id in self.histories:
            logger.info("Clearing history for session %s", session_id)
            del self.histories[session_id]