
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
import streamlit as st
st.set_page_config(page_title='Retrieval Augmented Generation')

api_key = "c09f91126e51468d88f57cb83a63ee36"
azure_endpoint = "https://chat-gpt-a1.openai.azure.com/"
api_version = "2023-03-15-preview"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="DanielChatGPT16k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key="c09f91126e51468d88f57cb83a63ee36",
    azure_endpoint="https://chat-gpt-a1.openai.azure.com/",
    api_version="2023-03-15-preview",
)

Settings.llm = llm
Settings.embed_model = embed_model


def rag(question,files):
    documents = SimpleDirectoryReader(input_files=files).load_data()

    node_parser = SentenceWindowNodeParser.from_defaults(window_size=3,window_metadata_key="window",original_text_metadata_key="original_text",)
    nodes = node_parser.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

    rerank = SentenceTransformerRerank(top_n = 2, model = "BAAI/bge-reranker-base")

    query_engine = index.as_query_engine(similarity_top_k = 6, alpha=0.5,node_postprocessors = [postproc, rerank],)
    response = query_engine.query(question)
    response = str(response)
    return response

uploaded_files = st.file_uploader("Choose a Document", accept_multiple_files=True)
files=[]
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    files.append('./data/'+uploaded_file.name)
print(files)
st.header('Retrieval Augmented Generation')
input=st.text_input('Input: ',key="input")
submit=st.button("Ask")

if submit:
    st.subheader("Implementing...")
    st.write(rag(input,files))