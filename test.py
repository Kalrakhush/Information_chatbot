import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
import os
from langchain.agents import ZeroShotAgent
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from PIL import Image
#1. Set up the tool
tools = FileManagementToolkit(
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()

read_tool, write_tool, list_tool = tools

#2. Set up the prompt and memory
prefix = """Have a conversation with a human, understand the full query and Answer the question correctly using correct sql queries mainly in aggregate functions and maintain history as the history of the messages is critical and very important to use. Also use full dataframe from the files provided not a part of it while answering the question. Check if the response can be represented in table then prefer that. Now, You have access to the following tools:"""
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
#3. Set up the LLM
# Securely load API key from Streamlit secrets (if available)
# api_key = st.secrets.get("OPENAI_API_KEY")

# # If API key not found in secrets, handle it gracefully
# if not api_key:
#     st.error("Please set your OpenAI API key in Streamlit secrets.")
# else:
#     # Set environment variable (optional, can be removed if unnecessary)
#     os.environ["OPENAI_API_KEY"] = """api_key""" 
# # llm_chain = LLMChain(llm=ChatOpenAI(temperature=0, model = "gpt-4"), prompt=prompt)
# os.environ["OPENAI_API_KEY"] = ""
multi_agent = create_csv_agent(
    ChatOpenAI(temperature=1.0, model="gpt-4"),
    ["Procurement.csv", "HCM_Data.csv","Finance.csv"],
    verbose=True,
    prompt=prompt,
    memory=memory,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

chain = load_qa_chain(multi_agent, chain_type="stuff")
# retriever = create_stuff_documents_chain(multi_agent)


# # Create chat chains
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# # history_aware_retriever = create_history_aware_retriever(
# #     multi_agent,
# #     retriever,
# #     contextualize_q_prompt
# # )

# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(multi_agent, qa_prompt)


# ### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory(session_id=session_id)
    return store[session_id]

agent_with_chat_history = RunnableWithMessageHistory(
    multi_agent,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: get_session_history(session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    handle_parsing_errors=True,
)
def generate_chart(data: pd.DataFrame):
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax)
    st.pyplot(fig)
# Function to query the chat bot for a response
# def query_chat_bot(question):
#     # Check if chain is initialized
#     if "chain" in globals() and hasattr(chain, "correct_method_name"):
#         # Use the question-answering chain if available
#         response = chain.correct_method_name(question)
#     else:
#         # Fall back to multi-agent if chain is not initialized
#         response = multi_agent.run(question)
#         return response

# # Define SessionState class for storing data
# class SessionState:
#     def __init__(self):
#         self.chat_history = []

st.set_page_config(page_title="Ask Me Information", page_icon="🔍")  # Set title and icon

col1, col2 = st.columns(2)
with col1:
    st.image(Image.open("info.jpg"), width=200)  # Replace with a bank logo image
with col2:
    st.title("Your Smart Informaton Assistant")

# Retrieve or generate session ID
session_id = st.text_input("Enter a session ID (or leave blank for a new session):")

# Chat history management (remains the same)
# Retrieve chat history for the session
session_history = get_session_history(session_id)

# Input field for user query with a clear prompt
query = st.text_input("What can I help you with today?", key="user_query_input")  # Use a unique key

response = ""
# Check if "Ask" button is clicked
# if st.button("Ask"):
#     if query:
#         # Invoke the chat chain to get the response
#         response = agent_with_chat_history.invoke(
#             {"input": query, "chat_history": session_history.messages},
#             config={"configurable": {"session_id": session_id}}
#         )["answer"]

#         # Add the user query and response to the chat history
#         session_history.add_message({"role": "user", "content": query})
#         session_history.add_message({"role": "ai", "content": response})

#         # Display the chatbot's response
#         st.write("**Chatbot Response:**")
#         st.write(response)
if st.button("Ask"):
    if query:
        # Invoke the chat chain to get the response
        response_dict = agent_with_chat_history.invoke(
            {"input": query, "chat_history": session_history.messages},
            config={"configurable": {"session_id": session_id}}
        )
        print("Response Dictionary:", response_dict)  # Print out the response dictionary

        # Extract the response from the dictionary
        response = response_dict.get("output", "No answer found")  # Get the value for the 'output' key
        print("Response:", response)  # Print out the extracted response

        # Add the user query and response to the chat history
        session_history.add_message({"role": "user", "content": query})
        session_history.add_message({"role": "ai", "content": response})

        # Display the chatbot's response
        st.write("**Chatbot Response:**")
        st.write(response)
        if 'chart_data' in response_dict:
            chart_data = response_dict['chart_data']
            chart_type = chart_data.get('type')
            data = chart_data.get('data')

            if chart_type and data:
                # Render the chart using st.pyplot()
                fig, ax = plt.subplots()
                if chart_type == 'bar':
                    ax.bar(data['x'], data['y'])
                elif chart_type == 'line':
                    ax.plot(data['x'], data['y'])
                elif chart_type == 'scatter':
                    ax.scatter(data['x'], data['y'])
                # Add more chart types as needed

                st.pyplot(fig)
                plt.close(fig)  # Close the figure to release memory
            else:
                st.warning("Invalid chart data format.")
            if 'chart_image_link' in response_dict:

                chart_image_link = response_dict['chart_image_link']
                st.image(chart_image_link) 