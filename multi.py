import os
import streamlit as st
import pandas as pd
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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from langchain.memory import ConversationBufferWindowMemory
from PIL import Image

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
# llm = ChatOpenAI()
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=600)
# prefix="You are working with csv files and when user ask a question always answer it after thinking. Maintain the context of questions asked to you. check if the asked question is amintaing the previous question's context"

# Create multi-agent for CSV data
multi_agent = create_csv_agent(
    ChatOpenAI(temperature=1.0, model="gpt-4-turbo"),
    ["Procurement.csv", "HCM_Data.csv","Finance.csv"],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

chain = load_qa_chain(multi_agent, chain_type="stuff")
# retriever = create_stuff_documents_chain(multi_agent)


# Create chat chains
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# history_aware_retriever = create_history_aware_retriever(
#     multi_agent,
#     retriever,
#     contextualize_q_prompt
# )

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(multi_agent, qa_prompt)

rag_chain = create_retrieval_chain(multi_agent, question_answer_chain)

### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


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


# Check if "Ask" button is clicked
if st.button("Ask"):
    if query:
        # Invoke the chat chain to get the response
        response = conversational_rag_chain.invoke(
            {"input": query, "chat_history": session_history.messages},
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        # Add the user query and response to the chat history
        session_history.add_message({"role": "user", "content": query})
        session_history.add_message({"role": "ai", "content": response})

        # Display the chatbot's response
        st.write("**Chatbot Response:**")
        st.write(response)
