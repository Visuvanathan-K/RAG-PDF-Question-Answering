import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import ChromaStorage
from custom_types import RAGChunksAndsrc,RAGQueryResult,RAGSearchResult,RAGUpsertResult
from groq import Groq


load_dotenv()

inngest_client = inngest.Inngest(
    app_id ="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()

)

@inngest_client.create_function(
    fn_id="Rag: Ingest PDF",
   trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle={
        "limit": 1,
        "period": datetime.timedelta(seconds=5),
        "burst": 2,
        "key": "event.data.source_id",
    },
    rate_limit={
        "limit": 10,
        "period": datetime.timedelta(hours=1),
        "key": "event.data.source_id",
    },
)

async def rag_inngest_pdf(ctx:inngest.Context):
    def _load(ctx:inngest.Context)-> RAGChunksAndsrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id",pdf_path)
        chunks=load_and_chunk_pdf(pdf_path)
        return RAGChunksAndsrc(chunks=chunks,source_id=source_id)



    def _upsert(chunks_and_src:RAGChunksAndsrc)->RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        ChromaStorage().upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk",lambda:_load(ctx),output_type=RAGChunksAndsrc)
    ingested=await ctx.step.run("upsert-and-embed",lambda:_upsert(chunks_and_src),output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int =5):
        query_vec = embed_texts([question])[0]
        store = ChromaStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided context."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": found.sources[:top_k],
        "num_contexts": len(found.contexts)
    }
    

app =FastAPI()




inngest.fast_api.serve(app,inngest_client,[rag_inngest_pdf,rag_query_pdf_ai])



