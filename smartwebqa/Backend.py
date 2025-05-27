from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

def load_docs(urls):
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from URLs.")
    return docs

def create_vector_store(docs, embedding_model_name):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Created FAISS vector store with embedded documents.")
    return vectorstore

def get_retriever(vectorstore, k=3):
    return vectorstore.as_retriever(search_kwargs={"k": k})

def answer_question(question, docs, qa_pipeline):
    # Combine text of top docs into one context
    context = " ".join([doc.page_content for doc in docs])
    if len(context.strip()) == 0:
        return "No context found to answer the question."
    
    result = qa_pipeline(question=question, context=context)
    return result.get('answer', 'No answer found.')

def main():
    # 1. URLs to load
    urls = [
            "https://en.wikipedia.org/wiki/Lionel_Messi "   ]
    
    # 2. Load documents from URLs
    docs = load_docs(urls)
    
    # 3. Create vector store with HF embeddings
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    vectorstore = create_vector_store(docs, embedding_model_name)
    
    # 4. Setup retriever
    retriever = get_retriever(vectorstore, k=3)
    
    # 5. Load Hugging Face question answering pipeline
    qa_model_name = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=qa_model_name)
    
    # 6. Ask questions in a loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        # Retrieve docs relevant to query
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Generate answer using reasoning model
        answer = answer_question(query, relevant_docs, qa_pipeline)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()

        
    
    
    