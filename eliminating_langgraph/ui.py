import gradio as gr
from app import faq_llm_call,lookup_knowledge_base,retriever
from prompt_faq import faq_text
from chat_history import ChatHistoryManager
from logger_config import setup_logger
from human_support import HumanSupportHandler
import traceback

# Set up loggers
main_logger = setup_logger('main')
retriever_logger = setup_logger('retriever')
llm_logger = setup_logger('llm')

# Initialize the history manager
history_manager = ChatHistoryManager()
human_support = HumanSupportHandler()

def respond(message, history):
    """Processes each message and returns the FAQ response."""
    session_id = "default_session"
    main_logger.info("Processing message for session %s", session_id)
    
    try:
        message = message.strip()
        
        # If already waiting for human support and message isn't 'stop'
        if human_support.waiting_for_human.get(session_id) and message.lower() != 'stop':
            return "Waiting for human support. Type 'stop' to resume normal chat."
            
        # Check for human support requests first
        support_response, should_process = human_support.handle_message(message, session_id)
        
        if support_response is not None:
            # Add the interaction to history
            history_manager.add_message(session_id, "user", message)
            history_manager.add_message(session_id, "assistant", support_response)
            return support_response
            
        if not should_process:
            return "Waiting for human support. Type 'stop' to resume normal chat."
            
        # Normal message processing
        history_manager.add_message(session_id, "user", message)
        chat_history = history_manager.format_history_for_prompt(session_id)
        relevant_context = lookup_knowledge_base(message, retriever)
        bot_message = faq_llm_call(relevant_context, message, chat_history)
        history_manager.add_message(session_id, "assistant", bot_message)
        
        return bot_message
        
    except Exception as e:
        main_logger.error("Error in respond function: %s\n%s", str(e), traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="SuperSupport",
    description="Feel free to ask anything related to SuperAGI",   
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)