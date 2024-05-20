from flask import Flask, render_template, request
import os
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Create multi-agent for CSV data
multi_agent = create_csv_agent(
    ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo"),
    ["Procurement.csv", "HCM_Data.csv","Finance.csv"],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

chain = load_qa_chain(multi_agent, chain_type="stuff")

# Function to query the chat bot for a response
def query_chat_bot(question, context=None):
    # Check if chain is initialized
    if "chain" in globals() and hasattr(chain, "correct_method_name"):
        # Use the question-answering chain if available
        response = chain.correct_method_name(question, context=context)
    else:
        # Fall back to multi-agent if chain is not initialized
        response = multi_agent.run(question)
    return response

# Define SessionState class for storing data
class SessionState:
    def __init__(self):
        self.chat_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    session_state = SessionState()

    if request.method == 'POST':
        prompt = request.form['prompt']
        if prompt:
            if session_state.chat_history:
                # If there is chat history, use the last question as context for follow-up
                last_question = session_state.chat_history[-1]['question']
                response = query_chat_bot(prompt, context=last_question)
            else:
                # If no chat history, just ask the question without context
                response = query_chat_bot(prompt)
            session_state.chat_history.append({"question": prompt, "response": response})
    else:
        prompt = None

    return render_template('index.html', prompt=prompt, chat_history=session_state.chat_history)

if __name__ == '__main__':
    app.run(debug=True)
