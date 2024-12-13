# app.py
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
import os

# Ensure you set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "KEY"

def load_faiss_index(faiss_index_file: str):
    """
    Load the FAISS index from the local file.

    Args:
        faiss_index_file (str): Path to the FAISS index file.

    Returns:
        FAISS: A loaded FAISS vector store instance.
    """
    return FAISS.load_local(faiss_index_file, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def create_retrieval_qa_chain(faiss_index):
    """
    Create a RetrievalQA chain using the FAISS index and OpenAI LLM.

    Args:
        faiss_index: The FAISS index object.

    Returns:
        RetrievalQA: A retrieval-based QA chain.
    """
    retriever = faiss_index.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

def main():
    """
    Main function to load the FAISS index, create the QA chain, and handle user queries.
    """
    # Load FAISS index
    faiss_index_file = "faiss_index"
    faiss_index = load_faiss_index(faiss_index_file)

    # Create QA chain
    qa_chain = create_retrieval_qa_chain(faiss_index)

    print("Welcome to the RAG-based Q&A system. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = qa_chain.run(query)
            print("\nAnswer:", response)
        except Exception as e:
            print("\nAn error occurred:", e)

if __name__ == "__main__":
    main()