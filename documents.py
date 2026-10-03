# simple_documents.py
# 12 simple documents (original text) + chunking.
# Use `chunks` for BM25 fit and for retriever.add_texts()

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# STEP 1: 12 simple documents
# ---------------------------------------------------------
documents = [
    # 1
    "Python is a popular programming language. It is easy to read and is used for web development, data science and automation. Python was created by Guido van Rossum.",
    # 2
    "A vector database stores embeddings. Pinecone is a vector database that can find the most similar vectors to a query very quickly.",
    # 3
    "BM25 is a keyword search algorithm. It gives a higher score to documents that contain the exact words from the query.",
    # 4
    "Embeddings turn text into numbers. Sentences with similar meaning get similar numbers, so a search for 'car' can also find 'automobile'.",
    # 5
    "Hybrid search combines keyword search and semantic search. It gives better results because it finds both exact words and similar meanings.",
    # 6
    "Gemini is a family of AI models from Google. It can answer questions, write text and help with coding tasks.",
    # 7
    "RAG means Retrieval Augmented Generation. First the system finds relevant documents, then the AI model uses them to write an accurate answer.",
    # 8
    "Ahmedabad is a city in Gujarat, India. It is known for its textile industry, the Sabarmati Riverfront and the old walled city, which is a UNESCO World Heritage site.",
    # 9
    "Drinking enough water is important for health. Most adults should drink around two to three liters of water per day, depending on weather and activity.",
    # 10
    "Regular exercise improves heart health and mood. Walking for thirty minutes a day is a simple way to stay active.",
    # 11
    "The solar system has eight planets. Jupiter is the largest planet, and Mercury is the closest planet to the Sun.",
    # 12
    "Cricket is a very popular sport in India. A cricket team has eleven players, and a match can be played in Test, ODI or T20 format.",
]

# ---------------------------------------------------------
# STEP 2: Split into chunks
# ---------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,      # max characters per chunk
    chunk_overlap=20,    # characters shared between neighbouring chunks
)

chunks = []
for doc in documents:
    chunks.extend(splitter.split_text(doc))

# ---------------------------------------------------------
# STEP 3: Test questions
# ---------------------------------------------------------
sample_queries = [
    "Who created Python?",                    # exact keyword
    "What is BM25?",                          # exact keyword
    "How does RAG work?",                     # concept
    "Which planet is the largest?",           # fact
    "How much water should I drink daily?",   # meaning
    "Why is combining two search methods better?",  # meaning (hybrid)
    "Tell me about Ahmedabad",                # keyword + meaning
]

if __name__ == "__main__":
    print("Original documents:", len(documents))
    print("Total chunks      :", len(chunks))
    print()
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(c)} chars): {c}")