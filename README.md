# RAG_MTG_Cards
Retrieve accurate information about Magic the Gathering cards through Retrieval Augmented Generation. 

### Vector Database
* Chunking of 5 cards at a time
* Embeddings performed on card names inside of the chunks, database containing index, card name, card text, and vector
* Embeddings using AnglE WhereIsAI/UAE-Large-V1
* When chunking 5 cards by full description and searching by name
    * 39.70% accuracy of exact card retrieval in top 3 chunks (15 cards)
    * 44.00% accuracy of exact card retrieval in top 5 chunks (25 cards)
    * Chunking embedding time ~15min
* When chunking 5 cards by name and searching by name
    * 57.9% accuracy of exact card retrieval in top 3 chunks (15 cards)
    * 64.4% accuracy of exact card retrieval in top 5 chunks (25 cards)
    * Chunking embedding time ~3min

### RAG
* Create embeddings using AnglE on the query input
* Perform semantic search between the query embeddings and each embedded vector in the vector database
* Retrieves highest similarity chunks for prompt addition
* Model prompt in instruction format including addition of highest similarity chunks
* ROUGE-L (Longest Common Subsequence) (0, 1) average score on 200 samples
    * RAG Model: 0.8821 
    * Baseline Model: 0.6353

### App
##### streamlit run app.py
* Generates original card text, text from RAG, and text from Baseline model