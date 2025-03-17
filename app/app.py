# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# import torch
# import os
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Path to FAISS index
# VECTOR_DB_PATH = "faiss_index"

# # Load embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Check if FAISS index exists, otherwise create it
# if os.path.exists(VECTOR_DB_PATH):
#     vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings)
# else:
#     # Load documents (update with your actual document path)
#     doc_path = "../about-nca.pdf"  # Change this to your document path
#     loader = PyMuPDFLoader(doc_path)
#     documents = loader.load()

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Create FAISS index
#     vector_store = FAISS.from_documents(texts, embeddings)
#     vector_store.save_local(VECTOR_DB_PATH)

# # Load the language model
# MODEL_NAME = "facebook/bart-large-cnn"
# device = 0 if torch.cuda.is_available() else -1  # Ensure device is an integer
# llm = HuggingFacePipeline.from_model_id(
#     model_id=MODEL_NAME,
#     task="text2text-generation",
#     device=device,
# )

# # Define prompt template
# prompt_template = """
#     You are an AI assistant. Answer the user's question using the provided context. 
#     If the context does not contain the answer, reply with "I don't know."
    
#     Context:
#     {context}
    
#     Question: {question}
#     Answer:
# """.strip()

# PROMPT = PromptTemplate.from_template(template=prompt_template)

# # Streamlit UI
# st.title("RAG-powered Chatbot")
# st.write("Ask a question, and I'll retrieve relevant documents and generate an answer!")

# user_query = st.text_input("Enter your question:")
# if user_query:
#     retrieved_docs = vector_store.similarity_search(user_query, k=3)


#     # Generate response
#     response = llm(prompt_template.format(context=" ".join([doc.page_content for doc in retrieved_docs]), question=user_query))

#     st.subheader("Response:")
#     st.write(response)

#     # Debugging: Print retrieved content
#     st.subheader("Retrieved Context:")
#     for doc in retrieved_docs:
#         st.text(doc.page_content[:300] + "...")  # Show first 300 chars for preview


#####################

# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# import torch
# import os
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Path to FAISS index
# VECTOR_DB_PATH = "faiss_index"

# # Load embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Check if FAISS index exists, otherwise create it
# if os.path.exists(VECTOR_DB_PATH):
#     vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings)
# else:
#     # Load documents (update with your actual document path)
#     doc_path = "about-nca.pdf"  # Change this to your document path
#     loader = PyMuPDFLoader(doc_path)
#     documents = loader.load()

#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Create FAISS index
#     vector_store = FAISS.from_documents(texts, embeddings)
#     vector_store.save_local(VECTOR_DB_PATH)

# # Load the language model
# MODEL_NAME = "lmsys/fastchat-t5-3b-v1.0"
# device = 0 if torch.cuda.is_available() else -1  # Ensure device is an integer
# llm = HuggingFacePipeline.from_model_id(
#     model_id=MODEL_NAME,
#     task="text2text-generation",
#     device=device,
#     model_kwargs={"max_length": 500, "temperature": 0.3, "do_sample": False}
# )

# # Define prompt template
# PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#     You are an AI assistant. Answer the user's question based on the provided context.
#     If the context does not contain the answer, reply with "I don't know."
    
#     Context:
#     {context}
    
#     Question: {question}
#     Answer:
#     """
# )

# # Streamlit UI
# st.title("RAG-powered Chatbot")
# st.write("Ask a question, and I'll retrieve relevant documents and generate an answer!")

# user_query = st.text_input("Enter your question:")
# if user_query:
#     # Retrieve relevant documents
#     retrieved_docs = vector_store.similarity_search(user_query, k=5)
#     retrieved_texts = "\n".join([doc.page_content[:500] for doc in retrieved_docs])  # Extract relevant text
    
#     # Debugging: Print retrieved content
#     st.subheader("Retrieved Context:")
#     for i, doc in enumerate(retrieved_docs):
#         st.text(f"Document {i+1}: {doc.page_content[:300]}...")
    
#     # Ensure we pass valid context to the model
#     if not retrieved_texts.strip():
#         response_text = "I don't know."
#     else:
#         # Generate response using formatted prompt
#         response_text = llm(PROMPT.format(context=retrieved_texts, question=user_query))
    
#     # Display generated response
#     st.subheader("Response:")
#     st.write(response_text)

#####################

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import torch
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to FAISS index
VECTOR_DB_PATH = "faiss_index"

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Check if FAISS index exists, otherwise create it
if os.path.exists(VECTOR_DB_PATH):
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings)
else:
    # Load documents (update with your actual document path)
    doc_path = "../about-nca.pdf"  # Change this to your document path
    loader = PyMuPDFLoader(doc_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create FAISS index
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

# Load the language model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = 0 if torch.cuda.is_available() else -1  # Ensure device is an integer  # Ensure device is an integer
llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text-generation",
    device=device,
    model_kwargs={"max_length": 100, "temperature": 0.3, "do_sample": True}
)

# Define prompt template
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant. Use the provided context to answer the user's question as accurately as possible.
    If the context does not contain the answer, respond with "I don't know." Do not make up an answer.
    
    Context:
    {context}
    
    User Question: {question}
    
    AI Answer: "
    """
)

# Streamlit UI
st.title("RAG-powered Chatbot")
st.write("Ask a question, and I'll retrieve relevant documents and generate an answer!")

user_query = st.text_input("Enter your question:")
if user_query:
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(user_query, k=3)
    retrieved_texts = ''.join([doc.page_content[:200] for doc in retrieved_docs[:2]]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content for doc in retrieved_docs])  # Extract relevant text
    
    # Debugging: Print retrieved content
    st.subheader("Retrieved Context:")
    for i, doc in enumerate(retrieved_docs):
        st.text(f"Document {i+1}: {doc.page_content[:300]}...")
    
    # Ensure we pass valid context to the model
    if not retrieved_texts.strip():
        response_text = "I don't know."
    else:
        # Generate response using formatted prompt
        formatted_prompt = PROMPT.format(context=retrieved_texts, question=user_query)
        response_text = llm.invoke(formatted_prompt)
    
    # Display generated response
    st.subheader("Response:")
    st.write(response_text)
