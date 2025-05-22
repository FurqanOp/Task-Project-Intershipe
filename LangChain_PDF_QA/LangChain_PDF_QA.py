import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import os

def main():
    st.set_page_config(page_title="PDF Q&A App", layout="wide")
    st.title("📄 PDF Question Answering App")

    st.markdown("""
        Welcome to the PDF Question Answering App!
        To get started, please upload a PDF document below.
        Once uploaded, you can ask questions about its content.
    """)

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        temp_file_path = "temp.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        try:
            st.info("🔄 Loading and processing PDF...")
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load_and_split()

            st.info("🧩 Splitting text into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pages)

            st.info("🧠 Generating embeddings...")
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectordb = FAISS.from_documents(docs, embedding)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            st.info("🧪 Loading FLAN-T5 model... This might take a moment.")
            llm_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,
                max_new_tokens=150,
                temperature=0.3,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True
            )
            llm = HuggingFacePipeline(pipeline=llm_pipeline)

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory
            )

            st.success("✅ PDF processed! Ask your question below.")
            user_question = st.text_input("Enter your question:")

            if user_question:
                with st.spinner("🔍 Searching for the answer..."):
                    response = qa_chain.invoke({"question": user_question})
                    st.markdown("### ✅ Answer:")
                    st.write(response["answer"])

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        st.info("Awaiting PDF upload. Please upload a PDF to begin.")

if __name__ == "__main__":
    main()
