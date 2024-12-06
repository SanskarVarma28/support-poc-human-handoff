import re
from dataclasses import dataclass
from typing import Optional, Tuple

class HumanSupportHandler:
    def __init__(self):
        self.waiting_for_human = {}  # session_id -> bool
        self.email_validation_mode = {}  # session_id -> bool
        self.user_emails = {}  # session_id -> email

    def validate_email(self, email: str) -> bool:
        """Validate email format using regex."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def handle_message(self, message: str, session_id: str) -> Tuple[str, bool]:
        """
        Handle incoming messages and manage support request states.
        Returns: (response_message, should_process_normally)
        """
        message = message.strip().lower()
        
        # Check if user wants to stop human support
        if message == "stop" and self.waiting_for_human.get(session_id):
            self.waiting_for_human[session_id] = False
            self.email_validation_mode[session_id] = False
            self.user_emails.pop(session_id, None)
            return "Resuming normal chat mode.", True

        # Check if in email validation mode
        if self.email_validation_mode.get(session_id):
            if self.validate_email(message):
                self.email_validation_mode[session_id] = False
                self.waiting_for_human.update({session_id: True})
                self.user_emails[session_id] = message
                return ("Thank you. Your email has been recorded and a human agent will contact you soon. "
                       "Type 'stop' to resume normal chat mode."), False
            else:
                return "Please provide a valid email address to continue:", False

        # Check for human support request
        human_request_phrases = [
            "talk to human", "speak to human", "connect to human", 
            "human support", "real person", "human agent"
        ]
        if any(phrase in message for phrase in human_request_phrases):
            self.email_validation_mode[session_id] = True
            return "Please provide your email address so we can connect you with a human agent:", False

        # Normal message processing
        return None, True