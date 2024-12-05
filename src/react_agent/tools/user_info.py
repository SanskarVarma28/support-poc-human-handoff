from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from react_agent.configuration import Configuration


@tool
def fetch_user_info(config: RunnableConfig) -> list[dict]:
    """Fetch the user details. Users are of three types: Visitors, Leads, and Account.
    Anyone who's email id is not known, will be classified as a visitor.
    If the email id is known, but the user does not have an acccount on the platform, then the user is classified a Lead.
    Otherwise, when the user has an account on our platform, it means he is using our products and then he is a Account.

    Returns:
        A list of dictionaries where each dictionary contains the user details like name, email id, and account id.
        When the user is a visitor, empty dictionary will be returned.
    """
    configurable = Configuration.from_runnable_config(config)
    user_info = {
        "email": configurable.email,
        "name": configurable.name,
        "account_id": configurable.account_id,
    }
    if not user_info:
        return {}
    else:
        return user_info
