import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


import pickle
from dotenv import load_dotenv

# Add a sidebar
with st.sidebar:
    st.title('Use LLM to Chat with your docs')
    st.subheader('😎')
    st.success('We are Live!', icon="✅")
    st.markdown('''
        ## About App
        App made with love by Gbenga and built using:
        
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(2)
    st.write('With 😍 by [Gbenga Ayelabola](https://gbengaayelab.github.io)')


def main():
    st.header("Extract Data & Information from Your PDFs 📚")
    load_dotenv()
    #Handle File Upload
    pdf = st.file_uploader("Upload your PDFs", type='pdf')
    
    
    if pdf is not None:
        # parse the pdf object into the instance of the PdfReader
        pdf_reader = PdfReader(pdf)
        # st.write(pdf.name) #name of pdf file
        # #test if the object is there
        # st.write(pdf_reader)
        text = ""
        for each_page in pdf_reader.pages:
            text += each_page.extract_text()
        
        # #check if it works
        # st.write(text)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200, # Keeps the contxt of the conversation
            length_function = len
            )      
        chunks = text_splitter.split_text(text=text)
       
       
      
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            # st.write('Embeddings Exists')
            # st.write(x)
        else:
             # create embedding object
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                vectorstore.save_local("vectorstore")


        # Interacting with users using prompting
        query = st.text_input("Start fetching info or data from your pdf file")
        # st.write(query) 

        if query:
            docs = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True).similarity_search(query=query, k=2)
            # st.write(docs)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Call the function to get the callback object
            callback = get_openai_callback()

            with callback as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()