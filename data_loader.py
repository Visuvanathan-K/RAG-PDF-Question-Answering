
from functools import lru_cache

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks


def embed_texts(texts: list[str]):
    if not texts:
        return []

    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
