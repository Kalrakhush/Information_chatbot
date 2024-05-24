from flask import Flask, render_template, request, jsonify
import os
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import ZeroShotAgent
from langchain.agents.agent_toolkits import FileManagementToolkit

app = Flask(__name__)

# Initialize tools
tools = FileManagementToolkit(
    selected_tools=["read_file", "write_file", "list_directory"]
).get_tools()

# Set up the prompt and memory
prefix = """Have a conversation with a human, understand the full query and Answer the question correctly using correct sql queries mainly in aggregate functions and maintain history as the history of the messages is critical and very important to use. Also use full dataframe from the files provided not a part of it while answering the question. Check if the response can be represented in table then prefer that. Now, You have access to the following tools:"""
suffix = """Begin!

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

memory = ChatMessageHistory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Set up the LLM
os.environ["OPENAI_API_KEY"] = ""

multi_agent = create_csv_agent(
    ChatOpenAI(temperature=1.0, model="gpt-4"),
    ["Procurement.csv", "HCM_Data.csv","Finance.csv"],
    verbose=True,
    prompt=prompt,
    memory=memory,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Function to manage session history
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory(session_id=session_id)
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    multi_agent,
    lambda session_id: get_session_history(session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    handle_parsing_errors=True,
)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/ask', methods=['POST'])
def ask():
    session_id = request.form.get('session_id', 'default_session')
    query = request.form['query']
    session_history = get_session_history(session_id)

    response_dict = agent_with_chat_history.invoke(
        {"input": query, "chat_history": session_history.messages},
        config={"configurable": {"session_id": session_id}}
    )

    response = response_dict.get("output", "No answer found")

    session_history.add_message({"role": "user", "content": query})
    session_history.add_message({"role": "ai", "content": response})

    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
