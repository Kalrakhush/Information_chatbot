import streamlit as st
from PIL import Image

from PyPDF2 import PdfReader
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
import pandas as pd

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)

# Load PDF files and combine text
pdfreader1 = pd.read_csv('Finance.csv')
# pdfreader2 = pd.read_csv('HCM_Data.csv')
# pdfreader3 = pd.read_csv('Procurement.csv')
# combined_data = pd.concat([pdfreader1,pdfreader2, pdfreader3],ignore_index=True)

# Clean data (remove missing values, outliers, etc.)
pdfreader1.dropna(inplace=True)
# pdfreader2.dropna(inplace=True)
# pdfreader1.dropna(inplace=True)

# Encode categorical variables
# pdfreader3 = pd.get_dummies(pdfreader3, columns=['Supplier Name', 'Item Name', 'Item Category'])
# pdfreader2 = pd.get_dummies(pdfreader2, columns=['Department', 'Nationality', 'Gender'])
pdfreader1 = pd.get_dummies(pdfreader1)

# Convert datetime features
# pdfreader3['PO Date'] = pd.to_datetime(pdfreader3['PO Date'], dayfirst=True)
# pdfreader1['Date of Expense'] = pd.to_datetime(pdfreader1['Date of Expense'])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(pdfreader1, embeddings)
retriever = vectorstore.as_retriever()
chain = load_qa_chain(llm, chain_type="stuff")

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

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
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Define API endpoint
@app.post('/api/chatbot')
def chatbot_api(query: str, session_id: str = ''):
    print('route accessed')
    session_history = get_session_history(session_id)

    # Invoke the chat chain to get the response
    response = conversational_rag_chain.invoke(
        {"input": query, "chat_history": session_history.messages},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    # Add the user query and response to the chat history
    session_history.add_message({"role": "user", "content": query})
    session_history.add_message({"role": "ai", "content": response})

    return {'response': response}

# Define Streamlit app
# st.title("PDF Chatbot")

# # Retrieve or generate session ID
# # session_id = st.text_input("Enter session ID:")

# # Retrieve chat history for the session
# session_history = get_session_history(session_id)

# # Input field for user query
# query = st.text_input("Enter your question:")

# # Check if "Ask" button is clicked
# if st.button("Ask"):
#     if query:
#         # Invoke the chat chain to get the response
#         response = conversational_rag_chain.invoke(
#             {"input": query, "chat_history": session_history.messages},
#             config={"configurable": {"session_id": session_id}}
#         )["answer"]
        
#         # Add the user query and response to the chat history
#         session_history.add_message({"role": "human", "content": query})
#         session_history.add_message({"role": "ai", "content": response})
        
#         # Display the chatbot's response
#         st.write("Chatbot Response:")
#         st.write(response)
    
#     # Display the updated chat history
#     st.write("Chat History:")
#     for chat_message in session_history.messages:
#         # Ensure that the message object is accessed correctly based on its type
#         role = chat_message.role if hasattr(chat_message, 'role') else "Unknown"
#         content = chat_message.content if hasattr(chat_message, 'content') else "Unknown"
#         if role != "Unknown":
#             st.write(f"{role.capitalize()}: {content}")
#     st.write("---")
# else:
#     st.write("Please enter a question.")


 
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

# # Display chat history with clear role and content separation
# chat_history_html = ""
# for message in session_history.messages:
#     role = getattr(message, 'role', None)
#     content = getattr(message, 'content', None)
#     if role == "user":
#         chat_history_html += f"""
#         <div style='margin-bottom: 10px;'>
#             <span style='font-weight: bold;'>{role.capitalize()}:</span> {content}
#         </div>
#         """
# st.markdown(chat_history_html, unsafe_allow_html=True)

#  # Display the chatbot's response
# st.write("**Chatbot Response:**")
# st.write(response)  
