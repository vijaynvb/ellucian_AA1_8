# business logic for chat with pdf 
from langchain_classic.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, BedrockEmbeddings   

load_dotenv()

# Load pdf 
def load_pdf(file_paths):
    text = ""
    for pdf in file_paths:
        loader = PdfReader(pdf)
        pages = loader.pages
        for page in pages:
            text += page.extract_text()
    return text

# split pages into chunks
def split_into_chunks(pdf_content, chunk_size=1000, chunk_overlap=200):
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(pdf_content)
    return chunks

# chunks into embeddings and store in vectorstore 
def create_vectorstore(chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# create chat model and retrieval chain 
# temperature is set to 0 for deterministic responses
def create_chat_chain(vectorstore):
    chat_model = ChatBedrock(model_id="mistral.mistral-7b-instruct-v0:2", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        chat_model,
        retriever,
        memory=memory
    )
    return qa_chain 


# LLM RAG -> we get different chunks -> you process the rerived data [chunks] with the help of llm -> Grammer, language 