from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma 

from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [
    "Vector databases enable semantic search by storing embeddings.",
    "RAG systems combine retrieval with language model generation.",
    "Embeddings capture semantic meaning in numerical form."
]

metadatas = [
    {"topic": "databases", "type": "technical"},
    {"topic": "AI", "type": "technical"},
    {"topic": "ML", "type": "technical"}
]

documents = [
    Document(page_content=text, metadata=metadatas[i])
    for i, text in enumerate(texts)
]

# 3. CORRECTED: Use the 'Chroma' class (not 'chorma')
vectordb = Chroma.from_documents( documents, embeddings )

# 4. CORRECTED: Use the 'vectordb' variable (not 'vectorstore')
results = vectordb.similarity_search_with_score("What is a RAG system?", k=2)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")