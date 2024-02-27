import os
import openai
import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_files():
    """Loads text from all .txt files in the data directory."""
    text = ""
    data_dir = os.path.join(os.getcwd(), "data")
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r") as f:
                text += f.read()
    return text


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text


def get_response(text):
    """Gets a summary of the given text using OpenAI's GPT-3."""
    prompt = f"""
    You are an expert in summarizing text. You will be given a text delimited by four backquotes,
    Make sure to capture the main points, key arguments, and any supporting evidence presented in the article.
    Your summary should be informative and well-structured, ideally consisting of 3-5 sentences.
    text: ''''{text}''''
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
    )
    return response["choices"][0]["message"]["content"]


def main():
    """Main function of the Streamlit app."""
    # Set page layout and title
    st.set_page_config(page_title="Summarizer", page_icon="")

    # Create a container with columns for better layout
    col1, col2 = st.columns([1, 2])  # Adjust column widths as needed

    # Display title in column 1
    with col1:
        st.title("Summarizer app")

    # Display app description in column 2
    with col2:
        st.write("This app uses OpenAI's GPT-3 to summarize a given text or a PDF file.")

    # Add a horizontal separator
    st.divider()

    # Choose input type radio button
    option = st.radio("Select Input Type", ("Text", "PDF"), key="input_type")

    if option == "Text":
        # Text input area
        user_input = st.text_area("Enter Text", "", key="text_input")

        # Submit button and response display
        if st.button("Submit", key="text_submit") and user_input != "":
            response = get_response(user_input)
            st.subheader("Summary")
            st.markdown(f"> {response}")
        else:
            st.error("Please enter text.")

    else:
        # PDF file uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_upload")

        # Submit button and response display
        if st.button("Submit", key="pdf_submit") and uploaded_file is not None:
            text = extract_text_from_pdf(uploaded_file)
            response = get_response(text=text)
            st.subheader("Summary")
            st.markdown(f"> {response}")
        else:
            st.error("Please upload a PDF file.")


if __name__ == "__main__":
    main()
