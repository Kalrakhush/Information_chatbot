from flask import Flask, render_template, request
import pandas as pd
import os
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Create multi-agent for CSV data
multi_agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    ["Procurement.csv", "HCM_Data.csv","Finance.csv"],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

chain = load_qa_chain(multi_agent, chain_type="stuff")

# Function to query the chat bot for a response
def query_chat_bot(question):
    # Check if chain is initialized
    if "chain" in globals() and hasattr(chain, "correct_method_name"):
        # Use the question-answering chain if available
        response = chain.correct_method_name(question)
    else:
        # Fall back to multi-agent if chain is not initialized
        response = multi_agent.run(question)
        return response

# Define SessionState class for storing data
class SessionState:
    def __init__(self):
        self.chat_history = []

session_state = SessionState()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        prompt = request.form["prompt"]
        if prompt:
            response = query_chat_bot(prompt)
            session_state.chat_history.append({"question": prompt, "response": response})
            return {"response": response}
        else:
            return {"response": "Please provide a prompt."}

if __name__ == "__main__":
    app.run(debug=True)
