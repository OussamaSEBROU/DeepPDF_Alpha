import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
import time

# Load environment variables and configure API
load_dotenv()
google_api_key = os.getenv("AIzaSyA557kuj8o_kSfQSmtYBUKFoX_7dvj_HiQ")
genai.configure(api_key=os.getenv("AIzaSyA557kuj8o_kSfQSmtYBUKFoX_7dvj_HiQ"))
os.environ["GOOGLE_API_KEY"] = "AIzaSyA557kuj8o_kSfQSmtYBUKFoX_7dvj_HiQ"


# Improved function to check if PDF is scanned
def is_scanned_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Check if the extracted text is mostly whitespace or very short
        if len(text.strip()) < 100:  # Adjust this threshold as needed
            return True
        return False
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return True  # Assume it's scanned if there's an error

# Function to process PDF
def process_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    return True

# Wisdom quotes
wisdom_quotes = [
    {"quote": "The only way to do great work is to love what you do.", "country": "United States"},
    {"quote": "Your time is limited, so don't waste it living someone else's life.", "country": "United States"},
    {"quote": "The best and most beautiful things in the world cannot be seen or even touched - they must be felt with the heart.", "country": "Germany"},
    {"quote": "Believe you can and you're halfway there.", "country": "United States"},
    {"quote": "Success is not final, failure is not fatal: it is the courage to continue that counts.", "country": "United Kingdom"},
    {"quote": "The journey of a thousand miles begins with a single step.", "country": "China"},
    {"quote": "The mind is everything. What you think you become.", "country": "India"},
    {"quote": "Life is what happens when you're busy making other plans.", "country": "United States"},
    {"quote": "Strive not to be a success, but rather to be of value.", "country": "United States"},
    {"quote": "The only limit to our realization of tomorrow will be our doubts of today.", "country": "United States"},
    {"quote": "The future belongs to those who believe in the beauty of their dreams.", "country": "Eleanor Roosevelt, United States"},
    {"quote": "It has become appallingly obvious that our technology has exceeded our humanity.", "country": "Albert Einstein, Germany"},
    {"quote": "The only person you are destined to become is the person you decide to be.", "country": "Ralph Waldo Emerson, United States"},
    {"quote": "The greatest glory in living lies not in never falling, but in rising every time we fall.", "country": "Nelson Mandela, South Africa"},
    {"quote": "Darkness cannot drive out darkness; only light can do that. Hate cannot drive out hate; only love can do that.", "country": "Martin Luther King Jr., United States"},
    {"quote": "The purpose of our lives is to be happy.", "country": "Dalai Lama, Tibet"},
    {"quote": "Our greatest weakness lies in giving up. The most certain way to succeed is always to try just one more time.", "country": "Thomas A. Edison, United States"},
    {"quote": "Life is not about finding yourself. Life is about creating yourself.", "country": "George Bernard Shaw, Ireland"},
    {"quote": "The best way to predict the future is to create it.", "country": "Abraham Lincoln, United States"},
    {"quote": "The only way to do great work is to love what you do.", "country": "Steve Jobs, United States"}
]

# Streamlit app setup
st.set_page_config("Chat PDF")
st.markdown("<h1 style='font-size: 3.5rem;'>DeepPDF Chat</h1>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 1.2rem; color: #00b9c6;'>Unlock the power of your PDFs with Large Language Models (LLMs) AI Technology. Ask insightful questions and receive precise, context-aware answers directly from your documents.</div>", unsafe_allow_html=True)
st.write("")

# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size: 1.6rem; font-weight: bold; color: #00b9c6;'>Interact with your PDFs</div>", unsafe_allow_html=True)

    pdf_docs = st.file_uploader("Upload your PDF File", accept_multiple_files=False, type="pdf")

    if st.button("Submit & Process"):
        if pdf_docs is not None:
            with st.spinner("Processing PDF..."):
                if is_scanned_pdf(pdf_docs):
                    st.error("The uploaded PDF appears to be scanned or has very little extractable text. Please upload an original PDF file.")
                else:
                    if process_pdf(pdf_docs):
                        st.success("PDF processed successfully!")
                        st.session_state.pdf_processed = True
        else:
            st.error("Please upload a PDF file.")

    # Help Section
    with st.expander("Help"):
        st.markdown("**Crafting Effective Prompts**")
        st.markdown("- **Specificity:** Formulate questions with precision. The more detailed your request, the more accurate and insightful the response will be.")
        st.markdown("- **Contextual Relevance:** Provide context to your questions. For instance, specify the document's subject matter or the information you seek.")
        st.markdown("- **Keyword Utilization:** Employ relevant keywords that appear in the PDF to guide DeepPDF Chat's understanding.")
        st.markdown("- **Question Variety:** Experiment with different question structures, such as 'who', 'what', 'when', 'where', 'why', or 'how', to explore various aspects of the document.")
        st.markdown("- **Multilingual Interaction:** DeepPDF Chat supports a wide range of languages. Feel free to ask questions in Arabic, French, Spanish, and more.")
        st.markdown("- **Language Translation:** DeepPDF Chat seamlessly translates text within the document, allowing you to ask questions in different languages.")
        st.markdown("- **Focus on Information Extraction:** Frame your questions to extract specific information, such as dates, names, or numbers.")

    # About Section
    with st.expander("About"):
        st.markdown("This app was developed by Mr. Oussama SEBROU")
        st.markdown("This app leverages DeepPDF Chat's AI for advanced PDF analysis.")
        st.markdown("- **DeepPDF Chat:** A powerful Large Language Model designed for complex tasks, including document understanding and question answering.")
        st.markdown("- **FAISS:** A fast library for efficient similarity search, enabling quick retrieval of relevant document chunks.")

# Main chat interface
if 'user_question' not in st.session_state:
    st.session_state.user_question = ''

user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", value=st.session_state.user_question)

if st.button("Submit Question"):
    if not hasattr(st.session_state, 'pdf_processed') or not st.session_state.pdf_processed:
        st.error("Please upload and process a PDF file before asking questions.")
    elif user_question:
        with st.spinner('Thinking...'):
            # Display a random wisdom quote while waiting
            random_quote = random.choice(wisdom_quotes)
            st.markdown(f"""<div style="font-size: 1.2rem; color: #888; font-style: italic;">"{random_quote["quote"]}" - {random_quote["country"]}</div>""", unsafe_allow_html=True)

            # Answer generation
            prompt_template = """
            You are an advanced document analysis and interaction AI your name is forever DeepPDF, specialized in multilingual translation, concise summarization, precise information extraction, and accurate question answering. Your responses are based solely on the provided text.

            **Document:**
            {context}

            **User Request:**
            {question}

            **Instructions:**
            1. **Analyze the Document:** Quickly scan the document to understand its structure, language, and main topics.
            2. **Interpret User Intent:** Determine if the user is requesting a translation, summary, information extraction, or question answering.
            3. **Execute the Request:** Perform the identified task with high accuracy and relevance to the document's content.
            4. **Format the Response:** Structure your output clearly, using appropriate headers and formatting for readability.
            5. **Ensure Accuracy:** Double-check your response against the original document to confirm factual correctness.
            6. **Handle Right-to-Left Languages:** If the output language is written from right to left (like Arabic), ensure the text is displayed correctly in the output.
            7. **If the output contains multiple languages, present them in the specified order, starting with the language indicated first by the user. Ensure proper formatting for each language as per its reading direction.
            8. **Enhance Detail and Accuracy:**
                - **Detailed Analysis:** Provide in-depth responses by exploring and integrating all relevant aspects of the document. Ensure that each part of the output is richly detailed and addresses all facets of the user's request.
                - **Precision and Depth:** Focus on the nuances and intricacies of the document. Highlight any subtleties, context-specific details, or complex elements that might be important for a comprehensive understanding.
                - **Professional Presentation:** Present your findings in a professional manner, with thorough explanations and justifications. Avoid generalizations and ensure that your output reflects a high level of expertise and careful consideration.
            9.  **Mathematical and Scientific Notation:** For any mathematical formulas, scientific symbols, or code snippets, present them like professional LaTeX font formatting for professional and accurate representation.
            10. **Emulate Authorial Voice:**  Capture the essence of the original author's writing style, tone, and voice in your response.  Mimic the original text's flow, phrasing, and vocabulary.  Do not introduce new information or deviate from the author's intent.
            11. **Output Length:** Your response should ideally be around 8000 tokens. 

            **Output:**
              <span style='font-size: 2.5rem; color: #00b9c6;font-weight: bold;'>[Your structured response based on the above format, tailored to the specific user request]</span>
            """

            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.markdown(response["output_text"], unsafe_allow_html=True)

            # Simulate thinking time (adjust as needed)
            time.sleep(2)
