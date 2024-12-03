import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pandasai.llm.local_llm import LocalLLM
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st
from bs4 import BeautifulSoup
import requests

# Load environment variables
load_dotenv()

# Initialize GROQ chat
groq_api_key = os.environ['GROQ_API_KEY']
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=0.2
)

# Function to process PDF files
def process_pdfs(files):
    texts = []
    metadatas = []
    for file in files:
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return chain

# Function to process website content
def process_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    website_text = soup.get_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(website_text)
    metadatas = [{"source": f"Website {i}"} for i in range(len(texts))]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return chain

# Function to chat with CSV data
def chat_with_csv(df, query):
    llm = LocalLLM(api_base="http://localhost:11434/v1", model="llama3")
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(query)
    return result

# Streamlit UI
st.set_page_config(layout='wide')
st.sidebar.markdown(
    """
    <style>
        .css-1y4m0w1 {
            margin-top: -50px;
        }
        .css-1wa0cu3 {
            padding-top: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.image(
    "./images/images/LOGOD1-removebg-preview.png",
    use_column_width=True,
    caption=None
)
st.title("Multi-file & Website ChatApp powered by LLM")
input_type = st.sidebar.radio("Select input type", ('PDF', 'CSV', 'Website URL'))

# Display icons for the selected input type
with st.sidebar.container():
    st.write("Preview of Selected Input:")
    if input_type == 'PDF':
        st.image(
            "./images/images/pdf.png",
            caption="PDF Input",
            width=120
        )
    elif input_type == 'CSV':
        st.image(
            "./images/images/csv.png",
            caption="CSV Input",
            width=120
        )
    elif input_type == 'Website URL':
        st.image(
            "./images/images/url.png",
            caption="Website URL Input",
            width=120
        )

if input_type == 'PDF':
    input_files = st.sidebar.file_uploader(
        "Upload your PDF files", type=['pdf'], accept_multiple_files=True, key="pdf_uploader"
    )
    if input_files:
        @st.cache_resource
        def cached_process_pdfs(files):
            return process_pdfs(files)
        
        chain = cached_process_pdfs(input_files)
        st.success(f"Processing {len(input_files)} PDF files done. You can now ask questions!")
        st.session_state.chain = chain

        if 'chain' in st.session_state:
            user_query = st.text_input("Ask a question about the PDFs:", key="pdf_query")
            if user_query:
                chain = st.session_state.chain
                res = chain.invoke(user_query)
                answer = res["answer"]
                st.write(answer)

elif input_type == 'CSV':
    input_files = st.sidebar.file_uploader(
        "Upload your CSV files", type=['csv'], accept_multiple_files=True, key="csv_uploader"
    )
    if input_files:
        selected_file = st.selectbox("Select a CSV file", [file.name for file in input_files])
        selected_index = [file.name for file in input_files].index(selected_file)
        data = pd.read_csv(input_files[selected_index])
        st.dataframe(data.head(3), use_container_width=True)
        input_text = st.text_area("Enter the query", key="csv_query")

        if input_text:
            if st.button("Chat with CSV"):
                result = chat_with_csv(data, input_text)
                st.success(result)

elif input_type == 'Website URL':
    url = st.text_input("Website URL", key="website_url")
    if url:
        @st.cache_resource
        def cached_process_website(url):
            return process_website(url)
        
        chain = cached_process_website(url)
        st.success("Website content processed successfully. You can now ask questions!")
        st.session_state.chain = chain

        if 'chain' in st.session_state:
            user_query = st.text_input("Ask a question about the website:", key="website_query")
            if user_query:
                chain = st.session_state.chain
                res = chain.invoke(user_query)
                answer = res["answer"]
                st.write(answer)
