import chromadb

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Vacuum the database
client.reset()
print("ChromaDB vacuuming completed!")
