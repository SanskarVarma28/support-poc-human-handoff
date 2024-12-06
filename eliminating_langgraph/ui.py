import gradio as gr
from app import faq_llm_call,lookup_knowledge_base,retriever,client
from prompt_faq import faq_text
from chat_history import ChatHistoryManager
from human_support import HumanSupportHandler
from logger_config import logger

# Initialize the history manager
history_manager = ChatHistoryManager()
human_support = HumanSupportHandler(client)

def respond(message, history):
    """Processes each message and returns the FAQ response."""
    session_id = "default_session"
    logger.info(f"""User entered the Query: {message}""")
    try:
        message = message.strip()
        
        # If already waiting for human support and message isn't 'stop'
        if human_support.waiting_for_human.get(session_id) and message.lower() != 'stop':
            return "Waiting for human support. Type 'stop' to resume normal chat."
            
        # Check for human support requests first
        support_response, should_process = human_support.handle_message(message, session_id)
        logger.info(f"""Support Response :{support_response}""")
        logger.info(f"""Human Escalation required :{not should_process}""")
        if support_response is not None:
            # Add the interaction to history
            history_manager.add_message(session_id, "user", message)
            history_manager.add_message(session_id, "assistant", support_response)
            return support_response
            
        if not should_process:
            return "Waiting for human support. Type 'stop' to resume normal chat."
            
        # Normal message processing
        history_manager.add_message(session_id, "user", message)
        logger.info(f"Chat history for this session is: {history_manager.get_history(session_id)}")
        chat_history = history_manager.format_history_for_prompt(session_id)
        relevant_context = lookup_knowledge_base(message, retriever)
        bot_message = faq_llm_call(relevant_context, message, chat_history)
        history_manager.add_message(session_id, "assistant", bot_message)
        
        return bot_message
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
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