
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import ChromaStorage

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Question Answering",
    page_icon="📄",
    layout="centered",
)

st.title("📄 RAG PDF Question Answering")
st.write("Upload a PDF and ask questions about its content.")

@st.cache_resource
def get_vector_store():
    return ChromaStorage(collection="demo_docs")

@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)

if "source_id" not in st.session_state:
    st.session_state.source_id = None

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    file_hash = hashlib.sha256(uploaded.getvalue()).hexdigest()

    if st.session_state.source_id != file_hash:
        if st.button("Process PDF"):
            temp_path = None
            try:
                with st.spinner("Extracting text and indexing PDF..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as temp_file:
                        temp_file.write(uploaded.getvalue())
                        temp_path = temp_file.name

                    chunks = load_and_chunk_pdf(temp_path)

                    if not chunks:
                        st.error("No extractable text was found in this PDF.")
                    else:
                        embeddings = embed_texts(chunks)
                        ids = [
                            hashlib.sha256(
                                f"{file_hash}:{i}".encode()
                            ).hexdigest()
                            for i in range(len(chunks))
                        ]
                        payloads = [
                            {
                                "source": uploaded.name,
                                "text": chunk,
                                "document_id": file_hash,
                            }
                            for chunk in chunks
                        ]

                        get_vector_store().upsert(ids, embeddings, payloads)
                        st.session_state.source_id = file_hash
                        st.session_state.source_name = uploaded.name
                        st.success(
                            f"Processed {uploaded.name}: {len(chunks)} chunks indexed."
                        )
            except Exception as exc:
                st.error(f"PDF processing failed: {exc}")
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink()

    else:
        st.success(f"PDF ready: {uploaded.name}")

st.divider()
st.subheader("Ask a question")

with st.form("question_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "Chunks to retrieve", min_value=1, max_value=10, value=5
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question.")
    elif not st.session_state.source_id:
        st.warning("Upload and process a PDF first.")
    else:
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                query_embedding = embed_texts([question.strip()])[0]

                results = get_vector_store().collection.query(
                    query_embeddings=[query_embedding],
                    n_results=int(top_k),
                    where={"document_id": st.session_state.source_id},
                )

                contexts = results.get("documents", [[]])[0]
                sources = results.get("metadatas", [[]])[0]

                if not contexts:
                    st.warning("No relevant text was found in this PDF.")
                else:
                    context_block = "\n\n".join(contexts)

                    response = get_groq_client().chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Answer using only the supplied PDF context. "
                                    "If the answer is not present, say so. "
                                    "Do not invent facts."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"PDF context:\n{context_block}\n\n"
                                    f"Question: {question.strip()}"
                                ),
                            },
                        ],
                        max_tokens=1024,
                    )

                    st.subheader("Answer")
                    st.write(response.choices[0].message.content)

                    st.caption("Source")
                    for metadata in sources:
                        st.write(f"- {metadata.get('source', 'Uploaded PDF')}")

        except Exception as exc:
            st.error(f"Question answering failed: {exc}")
