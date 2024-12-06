import re
from typing import Tuple
import openai
from logger_config import logger


class HumanSupportHandler:
    def __init__(self, oai_client):
        self.waiting_for_human = {}  # session_id -> bool
        self.email_validation_mode = {}  # session_id -> bool
        self.user_emails = {}  # session_id -> email
        self._client = oai_client
        logger.info("HumanSupportHandler initialized.")

    def detect_human_request(self, message: str) -> bool:
        """
        Use LLM to determine if the user is requesting human support.
        Returns True if human support is requested, False otherwise.
        """
        try:
            system_prompt = """Analyze if the user is requesting to speak with a human agent/representative.
            Return TRUE if:
            - User explicitly asks for human support/agent
            - User expresses desire to speak with a real person
            - User shows frustration with AI and wants human assistance
            - User mentions sales representative/team
            Return FALSE for all other cases.
            Respond with only 'true' or 'false'."""

            response = self._client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower() == 'true'
            return result

        except Exception as e:
            logger.error(f"Error in human request detection: {str(e)}")
            return False

    def validate_email(self, email: str) -> bool:
        """Validate email format using regex."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        is_valid = bool(re.match(pattern, email))
        if is_valid:
            logger.info(f"Valid email provided: {email}")
        else:
            logger.error(f"Invalid email provided: {email}")
        return is_valid

    def handle_message(self, message: str, session_id: str) -> Tuple[str, bool]:
        """
        Handle incoming messages and manage support request states.
        Returns: (response_message, should_process_normally)
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Check if user wants to stop human support
        if message_lower == "stop" and self.waiting_for_human.get(session_id):
            self.waiting_for_human[session_id] = False
            self.email_validation_mode[session_id] = False
            self.user_emails.pop(session_id, None)
            logger.info(f"User in session '{session_id}' resumed normal chat mode.")
            return "Resuming normal chat mode.", True

        # Check if in email validation mode
        if self.email_validation_mode.get(session_id):
            if message_lower == 'leave':
                self.email_validation_mode[session_id] = False
                logger.info(f"User in session '{session_id}' opted to continue with AI assistance.")
                return "Continuing with AI assistance.", True
                
            if self.validate_email(message):
                self.email_validation_mode[session_id] = False
                self.waiting_for_human[session_id] = True
                self.user_emails[session_id] = message
                return ("Thank you. Your email has been recorded and a human agent will contact you soon. "
                        "Type 'stop' to resume normal chat mode."), False
            else:
                return ("Please provide a valid email address so we can connect you with a human agent, "
                        "or type 'leave' to continue with AI assistance:"), False

        # Use LLM to detect human support request
        if self.detect_human_request(message):
            self.email_validation_mode[session_id] = True
            logger.info(f"User in session '{session_id}' requested human support.")
            return ("Please provide your email address so we can connect you with a human agent, "
                    "or type 'leave' to continue with AI assistance:"), False

        # Normal message processing
        return None, True