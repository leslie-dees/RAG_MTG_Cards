# RAG_MTG_Cards
Retrieve accurate information about Magic the Gathering cards through Retrieval Augmented Generation. 

### Vector Database
* Chunking of 5 cards at a time
* Embeddings performed on card names inside of the chunks, database containing index, card name, card text, and vector
* Embeddings using Ang1E WhereIsAI/UAE-Large-V1
* When chunking 5 cards by full description and searching by name
    * 39.70% accuracy of exact card retrieval in top 3 chunks (15 cards)
    * 44.00% accuracy of exact card retrieval in top 5 chunks (25 cards)
    * Chunking embedding time ~15min
* When chunking 5 cards by name and searching by name
    * 57.9% accuracy of exact card retrieval in top 3 chunks (15 cards)
    * 64.4% accuracy of exact card retrieval in top 5 chunks (25 cards)
    * Chunking embedding time ~3min