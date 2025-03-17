import streamlit as st
import os
import torch
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Set up API keys
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_key"
os.environ["GROQ_API_KEY"] = "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"

# Load FAISS-based vector store
def load_faiss():
    vector_path = "../vector-store"
    db_file_name = "nyein"
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-base"
    )
    vectordb = FAISS.load_local(
        folder_path=os.path.join(vector_path, db_file_name),
        embeddings=embedding_model,
        index_name="nca",
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever()

retriever = load_faiss()
groq_model = ChatGroq(model_name="llama-3.2-3b-preview", temperature=0.7)

# Define prompt template
prompt_template = """
You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Gentle & Informative Answer:
""".strip()

PROMPT = PromptTemplate.from_template(prompt_template)

def get_response(question):
    input_documents = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in input_documents])

    response = groq_model.invoke([{"role": "user", "content": PROMPT.format(context=context, question=question)}])
    
    return response.content, input_documents  # ✅ Use `.content` instead of `["text"]`

# Streamlit UI Setup
st.set_page_config(page_title="Chatbot with RAG", layout="centered", page_icon="💬")

# Custom CSS for better design
st.markdown("""
    <style>
        body {
            background-color: #F4F4F4;
        }
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
        }
        .stTextArea>div>textarea {
            font-size: 18px;
        }
        .stButton>button {
            font-size: 18px;
            padding: 8px 16px;
            background-color: #0084FF;
            color: white;
            border-radius: 8px;
        }
        .stMarkdown h2 {
            color: #0084FF;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.markdown("<h1 style='text-align: center;'>A6: Let’s Talk with Yourself", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'> Document-Supported Q&A</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Ask a question and get AI-powered responses based on retrieved documents.</p>", unsafe_allow_html=True)

# User Input Section
user_input = st.text_area("🔍 Ask a question:", "", height=100)

if st.button("Get Answer"):
    if user_input.strip():
        with st.spinner("🔎 Searching & Generating Response..."):
            response, documents = get_response(user_input)

        # Display chatbot response
        st.markdown("### 🤖 AI Response")
        st.info(response, icon="💡")

        # Display Supporting Documents (Collapsible)
        if documents:
            st.markdown("### 📄 Supporting Documents")
            for idx, doc in enumerate(documents):
                with st.expander(f"🔍 Document {idx + 1} (Click to Expand)"):
                    st.markdown(f"```{doc.page_content[:1000]}...```")  # Show first 1000 chars

    else:
        st.warning("⚠️ Please enter a question before clicking 'Get Answer'.")

# Footer
st.markdown("<br><p style='text-align: center; font-size: 14px; color: gray;'>Powered by Groq & FAISS | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)




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
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# device = 0 if torch.cuda.is_available() else -1  # Ensure device is an integer  # Ensure device is an integer
# llm = HuggingFacePipeline.from_model_id(
#     model_id=MODEL_NAME,
#     task="text-generation",
#     device=device,
#     model_kwargs={"max_length": 100, "temperature": 0.3, "do_sample": True}
# )

# # Define prompt template
# PROMPT = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#     You are an AI assistant. Use the provided context to answer the user's question as accurately as possible.
#     If the context does not contain the answer, respond with "I don't know." Do not make up an answer.
    
#     Context:
#     {context}
    
#     User Question: {question}
    
#     AI Answer:
#     """
# )

# # Streamlit UI
# st.title("RAG-powered Chatbot")
# st.write("Ask a question, and I'll retrieve relevant documents and generate an answer!")

# user_query = st.text_input("Enter your question:")
# if user_query:
#     # Retrieve relevant documents
#     retrieved_docs = vector_store.similarity_search(user_query, k=2)
#     retrieved_texts = ''.join([doc.page_content[:200] for doc in retrieved_docs[:2]]).join([doc.page_content[:200] for doc in retrieved_docs[:2]]).join([doc.page_content[:200] for doc in retrieved_docs[:2]]).join([doc.page_content[:200] for doc in retrieved_docs]).join([doc.page_content[:200] for doc in retrieved_docs[:2]]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content[:250] for doc in retrieved_docs]).join([doc.page_content for doc in retrieved_docs])  # Extract relevant text
    
#     # Debugging: Print retrieved content
#     st.subheader("Retrieved Context:")
#     for i, doc in enumerate(retrieved_docs):
#         st.text(f"Document {i+1}: {doc.page_content[:300]}...")
    
#     # Ensure we pass valid context to the model
#     if not retrieved_texts.strip():
#         response_text = "I don't know."
#     else:
#         # Generate response using formatted prompt
#         formatted_prompt = PROMPT.format(context=retrieved_texts, question=user_query)
#         response = llm.invoke(formatted_prompt)
#         response_text = response[0]['generated_text'] if isinstance(response, list) and 'generated_text' in response[0] else response
    
#     # Display generated response
#     st.subheader("Response:")
#     st.write(response_text)


# import streamlit as st
# import os
# import torch
# from langchain_groq import ChatGroq
# from langchain.vectorstores import FAISS
# from langchain import PromptTemplate
# from langchain.embeddings import HuggingFaceInstructEmbeddings




# # # Set Hugging Face API token
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your key" 

# os.environ["GROQ_API_KEY"] = "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"

# def load_faiss():
#     vector_path = "./vector-store"
#     db_file_name = "nlp_stanford"
#     embedding_model = HuggingFaceInstructEmbeddings(
#         model_name="hkunlp/instructor-base"
#     )
#     vectordb = FAISS.load_local(
#         folder_path=os.path.join(vector_path, db_file_name),
#         embeddings=embedding_model,
#         index_name="nlp",
#         allow_dangerous_deserialization=True
#     )
#     return vectordb.as_retriever()

# retriever = load_faiss()
# groq_model = ChatGroq(model_name="llama-3.3-70b-specdec", temperature=0.7)

# prompt_template = """
# Please answer the following question accurately based on the provided context of a person named Soe Htet Naing.
# Context:
# {context}

# Question: {question}

# Gentle & Informative Answer:
# """.strip()

# PROMPT = PromptTemplate.from_template(prompt_template)

# def get_response(question):
#     input_document = retriever.get_relevant_documents(question)
#     context = "\n".join([doc.page_content for doc in input_document])
#     response = groq_model.invoke(PROMPT.format(context=context, question=question))
#     return response, input_document

# # Streamlit UI Setup
# st.set_page_config(page_title="Chatbot with RAG", layout="wide")
# st.title("Chatbot - Document-Supported Q&A")
# st.write("Type your question below and get an AI-generated response with supporting sources.")

# user_input = st.text_input("Ask a question:", "")

# if user_input:
#     response, documents = get_response(user_input)
    
#     # Display chatbot response
#     st.subheader("AI Response")
#     st.write(response.content)
    
#     # Display supporting documents
#     st.subheader("Supporting Documents")
#     for doc in documents:
#         st.markdown(f"**Page Content:** {doc.page_content[:300]}...")


# import streamlit as st
# import os
# import torch
# from langchain_groq import ChatGroq
# from langchain.vectorstores import FAISS
# from langchain import PromptTemplate
# from langchain.embeddings import HuggingFaceInstructEmbeddings

# # Set up API keys (Ensure safety in production)
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_key"
# os.environ["GROQ_API_KEY"] = "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"

# # Load FAISS-based vector store
# def load_faiss():
#     vector_path = "../vector-store"
#     db_file_name = "nyein"
#     embedding_model = HuggingFaceInstructEmbeddings(
#         model_name="hkunlp/instructor-base"
#     )
#     vectordb = FAISS.load_local(
#         folder_path=os.path.join(vector_path, db_file_name),
#         embeddings=embedding_model,
#         index_name="nca",
#         allow_dangerous_deserialization=True
#     )
#     return vectordb.as_retriever()

# retriever = load_faiss()
# groq_model = ChatGroq(model_name="llama-3.2-3b-preview", temperature=0.7)

# # Define prompt template
# prompt_template = """
# You are a helpful AI assistant. Answer the question based on the provided context.

# Context:
# {context}

# Question: {question}

# Gentle & Informative Answer:
# """.strip()

# PROMPT = PromptTemplate.from_template(prompt_template)

# def get_response(question):
#     input_documents = retriever.get_relevant_documents(question)
#     context = "\n".join([doc.page_content for doc in input_documents])

#     response = groq_model.invoke([{"role": "user", "content": PROMPT.format(context=context, question=question)}])
    
#     return response.content, input_documents  # ✅ Use `.content` instead of `["text"]`


# # Streamlit UI Setup
# st.set_page_config(page_title="Chatbot with RAG", layout="wide")
# st.title("Chatbot - Document-Supported Q&A")
# st.write("Type your question below and get an AI-generated response with supporting sources.")

# # User input
# user_input = st.text_input("Ask a question:", "")

# if user_input:
#     response, documents = get_response(user_input)

#     # Display chatbot response
#     st.subheader("AI Response")
#     st.write(response)

#     # Display supporting documents
#     st.subheader("Supporting Documents")
#     for idx, doc in enumerate(documents):
#         st.markdown(f"**Document {idx + 1}:**\n```\n{doc.page_content[:500]}...\n```")

