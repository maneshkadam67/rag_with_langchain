import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -------------------------
# Load API Key
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Load Document
# -------------------------
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported format")

    return loader.load()

# -------------------------
# Create Vector Store
# -------------------------
def create_vectorstore(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# -------------------------
# Build RAG Chain
# -------------------------
def build_rag(vectorstore):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    file_path = "data/sample1.pdf"

    print("Loading document...")
    documents = load_document(file_path)

    print("Creating vector store...")
    vectorstore = create_vectorstore(documents)

    print("Building RAG pipeline...")
    rag_chain = build_rag(vectorstore)

    print("RAG Ready! Ask a question:")

    while True:
        query = input("\n>> ")
        if query.lower() == "exit":
            break

        result = rag_chain(query)

        print("\nAnswer:")
        print(result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata)