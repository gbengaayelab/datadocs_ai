import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Render custom CSS for Bootstrap
st.markdown("""
    <style>
        /* Add Bootstrap CSS */
        {% include 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css' %}
        
        /* Custom styling */
        .card {
            background-color: rgba(173, 216, 230, 0.2); /* Light blue with opacity */
            border: 1px solid rgba(173, 216, 230, 0.5);  /* Matching border color */
            backdrop-filter: blur(10px); 
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); 
        }
        
        .link-button {
            background-image: linear-gradient(to right, #56CCF2 0%, #2F80ED  51%, #56CCF2  100%);
            margin: 10px;
            padding: 15px 45px;
            text-align: center;
            text-transform: uppercase;
            transition: 0.5s;
            background-size: 200% auto;
            color: white;            
            box-shadow: 0 0 20px #eee;
            border-radius: 10px;
            display: block;
            text-decoration: none; 
        }

        .link-button:hover {
            background-position: right center; /* change the direction of the change here */
            color: #fff;
            text-decoration: none;         
        }
        
        .card-body {
            padding: 20px; 
        }
    </style>
""", unsafe_allow_html=True)
 
# Add a sidebar with Bootstrap styling
with st.sidebar:
    st.title('Start Chatting with your PDF 🧑‍🚀')
    st.success('We are Live!', icon="✅")
    st.markdown("""
        <div class="card">  
            <div class="card-body">
                <h5 class="card-title">Prompts Engineering Suggestions:</h5>
                <p class="card-text"><em>Share top tips on how to prepare for this role.</em></p>
                <p class="card-text"><em>Draft an email summarising this content for my boss.</em></p>
                <p class="card-text"><em>Compose a cover letter from this resume.</em></p>
                <p class="card-text"><em>Explain the contents in this book.</em></p>
                <p class="card-text"><em>Summarise this document in 10 bullet points.</em></p>
                <p class="card-text"><em>help answer the questions in this doc in a table.</em></p>
                <p class="card-text"><em>what are possible data protection and bridges that can result from this agreement?</em></p>
                
         
            
        </div>
        <a href= "https://gbengaayelab.github.io" style="text-decoration: none"/><div class="link-button">Visit My Website</div></a>
                
    """, unsafe_allow_html=True)



def main():
    st.header("Extract Data & Information from Your PDFs 📚")
    st.image(image='images\Jan-Work_2.jpg')
    
    # Handle File Upload
    pdf = st.file_uploader("Upload your PDFs", type='pdf')
    
    if pdf is not None:
        # Parse the PDF object into an instance of the PdfReader
        st.success("PDF fiule uploaded successfully", icon="✅")
        process_pdf(pdf)
        show_feedback_form(pdf)

        

def process_pdf(pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for each_page in pdf_reader.pages:
            text += each_page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # text will be split into chunks of maximum 1000 characters each
            chunk_overlap=200, # Keeps the context of the conversation
            length_function=len
        )      
        
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)   
        else:
            # Create embedding object
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                vectorstore.save_local("vectorstore")
            
        # Interacting with users using prompting
        query = st.text_input("Start fetching info or data from your pdf file")

        if query:
            docs= FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True).similarity_search(query=query,k=2)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")


            # Call the function to get the callback object
            callback = get_openai_callback()

            with callback as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
            print(cb)    #prints into terminal total cost
            
def show_feedback_form(pdf):    # Feedback Form
    
    st.subheader("Please provide your feedback on the response:")
   
    relevance_rating = st.slider("Relevance Rating:", min_value=1, max_value=5)
    accuracy_rating = st.slider("Accuracy Rating:", min_value=1, max_value=5)
    comments = st.text_area("Additional Comments:")
            
            # Submit Button
    if st.button("Submit Feedback"):
                # Save feedback to database or file
        save_feedback(relevance_rating, accuracy_rating, comments)
        st.success("Thank you for your feedback!")

def save_feedback(relevance_rating, accuracy_rating, comments):
        # Implement logic to save feedback to database or file
        # For demonstration purposes, print feedback to console
    print(f"Relevance Rating: {relevance_rating}")
    print(f"Accuracy Rating: {accuracy_rating}")
    print(f"Additional Comments: {comments}")

            # Footer
st.markdown("""
    <style>
                    /* Optional: Customize footer appearance if desired */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%; 
            background-color: #f0f8ff; /* Adjust as needed */ 
                    }
    </style>        
                        
    <footer class="footer mt-auto py-3 bg-light">
        <div class="container-fluid">
            <span class="text-muted">© 2024 Making Your Doc More Accessible</span>
        </div>
    </footer>
    """, unsafe_allow_html=True)    

if __name__ == '__main__':
    main()
