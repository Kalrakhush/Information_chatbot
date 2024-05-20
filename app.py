from flask import Flask, request, jsonify
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
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]= ""
llm = ChatOpenAI(model="gpt-4-turbo", temperature=1.0)

# # Get OpenAI API key from environment variables
# api_key = os.environ["OPENAI_API_KEY"]
# print("API Key:", api_key)  # Add this line for debugging

# Ensure API key is available
# if not api_key:
#     raise ValueError("Please set your OpenAI API key in the .env file.")

# Load PDF files and combine text
pdfreader1 = PdfReader('sustainability-16-01864-v2.pdf')
pdfreader2 = PdfReader('AI Adoption Strategy.pdf')
pdfreader3 = PdfReader('1-s2.0-S2199853123002469-main.pdf')
raw_text_1 = ''.join(page.extract_text() for page in pdfreader1.pages)
raw_text_2 = ''.join(page.extract_text() for page in pdfreader2.pages)
raw_text_3 = ''.join(page.extract_text() for page in pdfreader3.pages)
raw_text_combined = raw_text_1 + ' ' + raw_text_2 + ' ' + raw_text_3

# Split text
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
texts = text_splitter.split_text(raw_text_combined)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()

chain = load_qa_chain(llm, chain_type="stuff")

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

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    data = request.json
    query = data['query']
    session_id = data.get('session_id', '')
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": query, "chat_history": session_history.messages},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    session_history.add_message({"role": "user", "content": query})
    session_history.add_message({"role": "ai", "content": response})
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
