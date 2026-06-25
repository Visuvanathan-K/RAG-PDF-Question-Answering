from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for t in texts:
        chunks.extend(
            splitter.split_text(t)
        )

    return chunks


def embed_texts(texts: list[str]):
    embeddings = embed_model.encode(texts)
    return embeddings.tolist()