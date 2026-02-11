import operator
from typing import Annotated, Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class EmailState(
    BaseModel
):  # BaseModel is the Pydantic class for objects with annotated data fields
    # The email being processed
    email: Dict[str, Any]

    # Category of the email (inquiry, complaint, etc.)
    email_category: Optional[str] = None

    # Reason why the email was marked as spam
    spam_reason: Optional[str] = None

    # Analysis and decisions
    is_spam: bool = False

    # Response generation
    email_draft: Optional[str] = None

    # Using Annotated with operator.add allows LangGraph to merge
    # new message lists into the existing state automatically.
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)


# --- Logic & Nodes ---

model = ChatOpenAI(temperature=0)


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state.email
    # Here we might do some initial preprocessing
    print(f"Alfred is processing an email from {email['sender']}...")
    # No state changes needed here
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state.email
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    # Call the LLM
    response = model.invoke([HumanMessage(content=prompt)])
    assert isinstance(response.content, str), "Expected string response from LLM"
    response_text = response.content.lower()

    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    is_spam = "spam" in response_text and "not spam" not in response_text

    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": [
            response
        ],  # Annotated logic will append new messages automatically
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state.spam_reason}")
    print("The email has been moved to the spam folder.")

    # We're done processing this email
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state.email
    category = state.email_category or "general"

    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email["sender"]}
    Subject: {email["subject"]}
    Body: {email["body"]}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    return {"email_draft": response.content, "messages": [response]}


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg with the drafted response"""
    email = state.email
    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state.email_category}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state.email_draft)
    print("=" * 50 + "\n")
    # We're done processing this email
    return {}


def route_email(state: EmailState):
    """Determine the next step based on spam classification"""
    if state.is_spam:
        return "spam"
    else:
        return "legitimate"


# Create the graph
email_graph = StateGraph(EmailState)

# Add nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# Start the edges
email_graph.add_edge(START, "read_email")
# Add edges - defining the flow
email_graph.add_edge("read_email", "classify_email")

# Add conditional branching from classify_email
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {"spam": "handle_spam", "legitimate": "draft_response"},
)

# Add the final edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

compiled_graph = email_graph.compile()

# Example legitimate email
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith",
}

# Example spam email
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100.",
}

print("\nProcessing legitimate email...")
# Pydantic BaseModel requires me to pass an instance of EmailState
# I declared default values for all the fields except for email, so I am only passing the email content here
compiled_graph.invoke(EmailState(email=legitimate_email))

print("\nProcessing spam email...")
compiled_graph.invoke(EmailState(email=spam_email))
