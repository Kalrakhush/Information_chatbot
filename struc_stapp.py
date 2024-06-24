import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents import ZeroShotAgent
from langchain.agents.agent_toolkits import FileManagementToolkit
from PIL import Image
# Securely load API key from Streamlit secrets (if available)
api_key = st.secrets.get("OPENAI_API_KEY")

# If API key not found in secrets, handle it gracefully
if not api_key:
    st.error("Please set your OpenAI API key in Streamlit secrets.")
else:
    # Set environment variable (optional, can be removed if unnecessary)
    os.environ["OPENAI_API_KEY"] = api_key
# Ensure 'temp_files' directory exists
os.makedirs('temp_files', exist_ok=True)

# Set up the tool

tools = FileManagementToolkit(
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

# Set up the prompt and memory
prefix = """Have a conversation with a human, understand the full query and Answer the question correctly using correct sql queries mainly in aggregate functions and maintain history as the history of the messages is critical and very important to use. Also use full dataframe from the files provided not a part of it while answering the question. Check if the response can be represented in table then prefer that. And remember , you are able to show charts so explain them Now, You have access to the following tools:"""
suffix = """Begin!"

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

# Streamlit UI setup
st.set_page_config(page_title="Ask Me Information", page_icon="🔍")

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open("info.jpg"), width=200)  # Replace with a relevant image
with col2:
    st.title("Your Smart Information Assistant")

# File upload section
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

# Initialize an empty list to hold file paths
file_paths = []

# Check if files were uploaded
if uploaded_files:
    for file in uploaded_files:
        # Save the uploaded file temporarily
        file_path = os.path.join("temp_files", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        # Append the file path to the list
        file_paths.append(file_path)

# Create the agent with dynamically uploaded files
if file_paths:
    multi_agent = create_csv_agent(
        ChatOpenAI(temperature=1.0, model="gpt-4"),
        file_paths,
        verbose=True,
        prompt=prompt,
        memory=memory,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
    )

    # Load QA chain with the created agent
    chain = load_qa_chain(multi_agent, chain_type="stuff")

    # Store for session history
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
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

    # Retrieve or generate session ID
    session_id = st.text_input("Enter a session ID (or leave blank for a new session):")

    # Retrieve chat history for the session
    session_history = get_session_history(session_id)

    # Input field for user query with a clear prompt
    query = st.text_input("What can I help you with today?", key="user_query_input")

    chart_prioritisation = st.selectbox("Chart Prioritisation", ["Yes", "No"])

    if st.button("Ask"):
        if query:
            suffix = """. Give me answer both as text and also create a chart accordingly. While creating a chart, create a chart with axis names, and chart name. Provide charts with value labels. Also use diffrent types of charts when possible. You should always think about what to do. Also explain the chart using text."""

            mixed_output = []
            full_query = query + suffix if chart_prioritisation == "Yes" else query

            # Run the prompt through the agent
            # Invoke the chat chain to get the response
            response_dict = agent_with_chat_history.invoke(
                {"input": full_query, "chat_history": session_history.messages},
                config={"configurable": {"session_id": session_id}}
            )
            response = response_dict.get("output", "No answer found")
            session_history.add_message({"role": "user", "content": query})
            session_history.add_message({"role": "ai", "content": response})

            st.write("**Chatbot Response:**")
            st.write(response)

            # Attempt to display the chart if present
            try:
                plot_buffer = BytesIO()
                plt.savefig(plot_buffer, format='png', bbox_inches='tight')
                plot_buffer.seek(0)
                mixed_output.append(plot_buffer.getvalue())
                st.image(mixed_output[0], use_column_width=True)
            except:
                pass

# Clean up temporary files (if necessary) at the end of the session
for file_path in file_paths:
    os.remove(file_path)
