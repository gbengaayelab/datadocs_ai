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

# Add custom CSS for Bootstrap
st.markdown("""
    <style>
        /* Add Bootstrap CSS */
        {% include 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css' %}
        .card {
            background-color: rgba(173, 216, 230, 0.2); /* Light blue with opacity */
            border: 1px solid rgba(173, 216, 230, 0.5);  /* Matching border color */
            backdrop-filter: blur(10px); 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); 
        }
        .link-button {
            background-color: rgba(173, 216, 230, 0.8); /* Example: More opaque button */
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            border-radius: 8px; 
            cursor: pointer;
        }
        .card-body {
            padding: 20px; 
        }
    </style>
""", unsafe_allow_html=True)

# Add a sidebar with Bootstrap styling
with st.sidebar:
    st.title('Start Chating with your pdf🧑‍🚀')
    st.success('We are Live!', icon="✅")
    st.markdown("""
<div class="card">  
    <div class="card-body">
        <h5 class="card-title">Streamlit</h5>
        <p class="card-text">For building the web interface.</p>
    </div>
</div>

<div class="card"> 
    <div class="card-body">
        <h5 class="card-title">LangChain</h5>
        <p class="card-text">For interacting with LLMs.</p>
    </div>
</div>

<div class="card"> 
    <div class="card-body">
        <h5 class="card-title">OpenAI</h5>
        <p class="card-text">For the LLM power.</p>
    </div>
</div>
<div class="card"> 
    <div class="card-body" id="author-link">
        <h5 class="card-title">With 😍 by Gbenga Ayelabola</h5>
        <a href="https://gbengaayelab.github.io" target="_blank" class="link-button">Visit My Website</a>
    </div>
</div>
                
<script>
    document.getElementById("author-link").addEventListener("click", function() {
        window.open("https://gbengaayelab.github.io", '_blank'); 
    });
</script>
""", unsafe_allow_html=True)

def main():
    st.header("Extract Data & Information from Your PDFs 📚")
    st.image(image='images\Jan-Work_2.jpg')
    
    load_dotenv()
    # Handle File Upload
    pdf = st.file_uploader("Upload your PDFs", type='pdf')
    
    if pdf is not None:
        # parse the pdf object into the instance of the PdfReader
        pdf_reader = PdfReader(pdf)
        text = ""
        for each_page in pdf_reader.pages:
            text += each_page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # Keeps the context of the conversation
            length_function=len
        )      

        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            # create embedding object
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                vectorstore.save_local("vectorstore")

        # Interacting with users using prompting
        query = st.text_input("Start fetching info or data from your pdf file")

        if query:
            docs = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True).similarity_search(query=query, k=2)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # Call the function to get the callback object
            callback = get_openai_callback()

            with callback as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
     # Footer
    st.markdown("""
    <style>
        /* Optional: Customize footer appearance if desired */
        .footer {
            position: fixed;
            bottom: 0;
            width: 200%; 
            background-color: #f0f8ff; /* Adjust as needed */ 
        }
    </style> 
    <footer class="footer mt-auto py-3 bg-light">
        <div class="container-fluid">
            <span class="text-muted">       © 2024 Making Your Doc More Accessible                   </span>
        </div>
    </footer>
    """, unsafe_allow_html=True)    
if __name__ == '__main__':
    main()
