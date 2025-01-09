import streamlit as st


# Load environment variables
load_dotenv()

# Set up API key for Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract text from uploaded PDFs
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store using HuggingFace embeddings
def create_and_save_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # HuggingFace embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain using Gemini API
def create_prompt_template():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer contains any structured data like tables or lists, respond in the same format. 
    If the answer is not in the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return prompt 

# Function to handle user input and provide a response
def handle_user_query(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Same HuggingFace embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])  # Combine the documents for context

    prompt = create_prompt_template()
    formatted_prompt = prompt.format(context=context, question=user_question)

    # Call Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(formatted_prompt)
    
    st.write("Reply: ", response.text if response.text else "No response generated.")

# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    user_question = st.text_input("Ask a relevant Question")

    if user_question:
        handle_user_query(user_question)

    with st.sidebar:
        st.title("Upload PDF 📂")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                raw_text = extract_pdf_text(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                create_and_save_vector_store(text_chunks)
                st.success("Processing Done")

if __name__ == "__main__":
    main()
