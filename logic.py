from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN"), 
)

# get PDF document and extract text
def get_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return None


def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


def vector_db(chunks):
    """Create a vector database."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None


def user_question(vectorstore, user_input):
    """Generate an answer to the user's question."""
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        # docs = retriever.get_relevant_documents(user_input)
        docs = retriever.invoke(user_input)

        # Extract context from retrieved documents
        context = "\n".join([doc.page_content for doc in docs])

        # Define the prompt
        prompt = f"""
        Given the following context and a question, generate an answer based on this context only.
        Use as much text as possible from the "response" section in the source document context without making major changes.
        If the answer is not found in the context, state: "Your request is beyond my scope of assessment."
        
        CONTEXT: {context}
        QUESTION: {user_input}
        """

        # Generate response using the OpenAI client
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in QA generation: {str(e)}")
        return "An error occurred while generating the response."


if __name__ == "__main__":
    # Load and process the PDF
    pdf_path = input("Enter the path to your PDF file: ")
    text = get_pdf_text(pdf_path)
    
    if text:
        print("PDF text successfully extracted.")
        chunks = get_text_chunks(text)
        vectorstore = vector_db(chunks)
        
        if vectorstore:
            print("Vector database created successfully.")
            
            while True:
                # Get user question
                user_input = input("\nAsk a question (type 'exit' to quit): ")
                if user_input.lower() == "exit":
                    print("Exiting...")
                    break
                
                # Get response
                response = user_question(client, vectorstore, user_input)
                print(f"Response: {response}")
        else:
            print("Failed to create vector database.")
    else:
        print("Failed to extract text from the PDF.")
