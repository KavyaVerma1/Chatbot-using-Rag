import streamlit as st
import pinecone
import os
import time
from pinecone import ServerlessSpec
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from groq import Groq

# Streamlit setup
st.title("Harry Potter Vector Search with Streamlit")

# Initialize Pinecone and Hugging Face embeddings
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_field = "text"

# Set up Pinecone and Groq API keys from secrets
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not pinecone_api_key or not groq_api_key:
    st.error("API keys for Pinecone and Groq are missing. Please set them in secrets.toml.")
    st.stop()

# Setup Pinecone connection
cloud = os.environ.get("PINECONE_CLOUD", "aws")
region = os.environ.get("PINECONE_REGION", "us-east-1")
spec = ServerlessSpec(cloud=cloud, region=region)

pc = pinecone.Pinecone(api_key=pinecone_api_key)
index_name = "harry-potter"

# Initialize Pinecone index when button is clicked
if st.button("Initialize Pinecone Index"):
    try:
        # Delete the index if it already exists
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)

        # Create a new index
        pc.create_index(index_name, dimension=768, metric="dotproduct", spec=spec)

        # Wait for index initialization
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        st.success("Pinecone index initialized successfully!")

    except Exception as e:
        st.error(f"An error occurred during Pinecone initialization: {e}")

# Create the vector store with the Pinecone index
index = pc.Index(index_name)
vectorstore = Pinecone(index, embed.embed_query, text_field)
st.success("Vectorstore initialized successfully.")

# Setup Groq
client = Groq(api_key=groq_api_key)
llm = ChatGroq(temperature=0.2, model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

# Setup RetrievalQA with the vectorstore as retriever
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

# User input for query
st.header("Ask Questions from Harry Potter Books")
user_query = st.text_input("Enter your question here:")

if user_query:
    st.write("Searching for answers...")
    try:
        # Query the model and get the answer
        answer = qa.run(user_query)
        st.write(f"**Answer:** {answer}")

    except Exception as e:
        st.error(f"An error occurred while processing the query: {e}")
