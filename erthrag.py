import streamlit as st
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')

st.image("Erth.png", use_column_width=True)
st.title("Erth | إرث")

# Set API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Initialize OpenAI model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

# Set up the prompt template
template = """
Anwser the questions based on the context below in arabic.
the context below having information about Saudi Arabia's Culture, heritage, and historical sites.
Do not mention the context explicitly in your answer ever.
If you can't anwser the question, reply "I don't know".

Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

# Load data
data_path = "data.txt"
loader = TextLoader(data_path)
text = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splitted = text_splitter.split_documents(text)

# Embedding
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "erth"

# Pinecone Vector Store
pinecone_vectorstore = PineconeVectorStore.from_documents(
    splitted, embedding, index_name=index_name
)

# Final chain with Pinecone
chain = (
    {"context": pinecone_vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

# Tabs
tab1 = st.tabs(["FT-AraGPT2 Text-to-text"])

with tab1:
    st.header("Fine-Tuned AraGPT2 Text-To-Text")
    if "gpt2_messages" not in st.session_state:
        st.session_state.gpt2_messages = []

    for message in st.session_state.gpt2_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("إسألني عن التراث السعودي (AraGPT2)"):
        st.session_state.gpt2_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chain.invoke(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.gpt2_messages.append({"role": "assistant", "content": response})
