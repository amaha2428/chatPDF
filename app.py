import streamlit as st
from logic import get_pdf_text, get_text_chunks, vector_db, user_question

# Header for the application
st.header("Chat with Multiple PDFs")

# Input box for user question
question = st.text_input('Ask a question about your document:')

# Initialize session state for the vector database
if 'vector' not in st.session_state:
    st.session_state.vector = None

# Process user question when the vector database is ready
if question and st.session_state.vector is not None:
    # Generate a response for the user's question
    response = user_question(vectorstore=st.session_state.vector, user_input=question)
    st.write(f"**Response:** {response}")
elif question and st.session_state.vector is None:
    st.warning("Please upload and process the PDF files first.")

# Sidebar for uploading and processing PDFs
with st.sidebar:
    st.subheader("Instructions:")
    st.write("""
        1. Upload PDF files by clicking the 'Browse files' button below.\n
        2. Click on the 'Process' button to train the model on the uploaded PDFs.\n
        3. Once the model is trained, ask questions about the content of the PDFs.
    """)

    # File uploader for PDFs
    pdf_files = st.file_uploader(
        'Upload PDF files here (multiple files allowed):',
        accept_multiple_files=True,
        type=['pdf']
    )

    # Process button to create the vector database
    if st.button("Process"):
        if pdf_files:
            with st.spinner("Processing PDFs..."):
                try:
                    # Extract text from uploaded PDFs
                    raw_text = ""
                    for pdf_file in pdf_files:
                        raw_text += get_pdf_text(pdf_file)

                    # Create text chunks and vector database
                    chunks = get_text_chunks(raw_text)
                    vector = vector_db(chunks)
                    st.session_state.vector = vector
                    st.success("Vector database built successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload at least one PDF file to process.")
