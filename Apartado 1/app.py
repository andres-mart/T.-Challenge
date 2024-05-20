import os
import io
import PyPDF2
import streamlit as st
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai

# Text extraction function
def extract_pdf(pdf):
    pdf_bytes = io.BytesIO(pdf)
    pdf_reader = PyPDF2.PdfReader(pdf_bytes)
    num_pages = len(pdf_reader.pages)
    detected_text = ''

    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num] 
        detected_text += page_obj.extract_text() + '\n\n' # Considering 2 paragraphs

    pdf_bytes.close()
    return detected_text

# Division of the text into paragraphs
def get_paragraphs_by_page(detected_text):
    text_by_page = {}
    paragraphs = detected_text.split('\n\n')
    for page_num, paragraph in enumerate(paragraphs, start=1):
        if page_num not in text_by_page.keys():
            text_by_page[page_num] = [paragraph]
        else:
            text_by_page[page_num].append(paragraph)

    return text_by_page

# Text tokenization
def split_texts(texts, chunk_size = 200, chunk_overlap=20):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(texts)

    return docs
# Number of tokens and overlap by dividing by tokens
chunk_size = 200
chunk_overlap = 20

# Text chunking and metadata
def create_chunks_and_metadata(text_by_page, document):
    preprocess_text = []
    for key in text_by_page:
        text = "\n".join(text_by_page[key])
        preprocess_text.append({"text": text, "page": key, "document": document})
    return preprocess_text

index_name = "index_list"
chunks_path = "chunks_data"

# Credentials and definition of the response engine
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv("DEPLOYMENT_NAME")

def get_completion(prompt, model="model_name"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# Summary functions
def count_tokens(text):
    return len(text.split())

def summarize_conversation(conversation):
    prompt = f"Summarize the following conversation: {conversation}"
    summary = get_completion(prompt)
    return summary

def main():
    
    st.title("Extractor de texto de PDF")
    archivo_pdf = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
    user_query = st.text_input("Ask a question about the uploaded PDFs")
    submit_button = st.button("Submit")

    if submit_button:
        if archivo_pdf is not None:
            chunks_list = []
            for file in archivo_pdf:
                texto_extraido = extract_pdf(file.read())
                paragraphs_by_page = get_paragraphs_by_page(texto_extraido)
                document_name = file.name
                preprocess_text = create_chunks_and_metadata(paragraphs_by_page, document_name)
                chunks_list.extend(preprocess_text)

            df_chunks = pd.DataFrame(chunks_list)
            df_chunks.to_csv(f"{chunks_path}/chunks.csv", sep=",", index=False)
            loader = DataFrameLoader(pd.DataFrame(chunks_list), page_content_column="text")
            data = loader.load()
            document_chunks = split_texts(texts=data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # st.write(document_chunks)

            # Vector base creation
            embeddings = OpenAIEmbeddings(deployment=os.getenv('OPENAI_EMBEDDINGS_DEPLOYMENT'))
            db = FAISS.from_documents(document_chunks, embeddings)
            db.save_local(index_name)
            db = FAISS.load_local(index_name, embeddings)

            similarity_docs = db.similarity_search(query=user_query, k=3)

            # Prompt definition and response
            prompt = f""""
            You are an effective helper, given the context information you must answer the user's questions in a clear and concise way.

            Context:
            {similarity_docs}

            Question: {user_query}
            """

            response = get_completion(prompt)
            st.write(response)

            # The conversation is summarized to 200 tokens:
            conversation = []
            total_tokens = 0
            total_tokens += count_tokens(user_query) + count_tokens(response)

            if total_tokens > 200:
                summarized_conversation = summarize_conversation(" ".join(conversation))
                # Reset conversation:
                conversation = [summarized_conversation]
                total_tokens = count_tokens(summarized_conversation)
                st.write("Conversation so far:")
                st.write("\n".join(conversation))

            st.write(total_tokens)

if __name__ == "__main__":
    main()
