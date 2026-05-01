RAG-Question Answering System Using Dense Sentence Embeddings


-----------------
This program implements a Retrieval-Augmented Generation (RAG) pipeline for open-domain question answering.
It retrieves relevant passages from Wikipedia using dense sentence embeddings and cosine similarity, then passes the top passages to an LLM to generate a concise, grounded answer.
The goal is to show that a language model's answers can be improved and made more factual by grounding them in retrieved evidence rather than relying on the model's parametric memory alone.

This code runs in terminal:
1. Prompts the user to enter a question.
2. Extracts keywords from the question using an LLM, then retrieves the first valid Wikipedia article from the search results.
3. Splits the article into 3-sentence chunks and ranks them by embedding cosine similarity to the question.
4. Sends the top 3 chunks along with the question to an LLM and prints the generated answer.
5. Prints the top 1-3 evidence passages that were used to generate the answer.
-----------------
Usage Instructions:

The program runs from the commandline with no arguments:
python rag-qa.py
Then, The user is prompted to type their question directly in the terminal after that. 
The program prints the answer followed by the supporting evidence passages.

-----------------
External Dependencies:
   - wikipedia: fetches and parses Wikipedia article content
     Install with: pip install wikipedia
   - sentence-transformers: dense sentence embeddings via all-MiniLM-L6-v2
     Install with: pip install sentence-transformers
   - sklearn (scikit-learn): cosine_similarity
     Install with: pip install scikit-learn
   - groq: client for the Groq-hosted LLaMA 3.1 inference API
     Install with: pip install groq
   - python-dotenv: loads the GROQ_API_KEY from a .env file
     Install with: pip install python-dotenv
   - Standard library: os, re

-----------------
Example Interaction:
Enter your question: What is the capital of Japan?
Answer: The capital of Japan is Tokyo.
Evidence:
1. The capital of Japan is Tokyo. Throughout history, the national capital of Japan has been in locations other than Tokyo. The oldest capital is Nara.
2. (2003). Japanese Capitals in Historical Perspective: Place, Power and Memory in Kyoto, Edo and Tokyo. New York: Psychology Press.
3. This was the first time that a central government office has been relocated outside Tokyo since Tokyo was designated as the capital. 
    This list of legendary capitals of Japan begins with the reign of Emperor Jimmu. The names of the Imperial palaces are in parentheses: 
    Kashihara, Yamato at the foot of Mount Unebi during reign of Emperor Jimmu Kazuraki, Yamato during reign of Emperor Suizei Katashiha, 
    Kawachi during the reign of Emperor Annei Karu, Yamato during reign of Emperor Itoku.

-----------------
System Design:

1. Retrieval
   Wikipedia is used as the knowledge source.
   An LLM first extracts 3-5 search keywords from the user's question to form a cleaner Wikipedia query.
   The Wikipedia article content is split into chunks of 3 sentences each.
   Both the chunks and the user question are encoded into dense vectors using the all-MiniLM-L6-v2 sentence embedding model.
   Cosine similarity between the question embedding and each chunk embedding is computed to rank relevance.
   Chunks with a similarity score below 0.2 are filtered out as noise.
   The top 3 most relevant chunks are returned as evidence.

2. Answer Generation
   The top-k retrieved chunks are concatenated into a context string.
   The question and context are formatted into a prompt and sent to LLaMA 3.1 (8B) via the Groq API.
   The model is instructed to answer concisely in one sentence using only the provided context.
   If the answer cannot be found in the context, the model responds that it does not know.

3. Evidence
   The top retrieved passages are printed with numbering after the answer to show what the model's response is grounded in.

-----------------
Design Choices:

Sentence embeddings (all-MiniLM-L6-v2 via sentence-transformers) replace TF-IDF for retrieval because they capture semantic meaning rather than just lexical overlap.
This allows the system to match chunks that are topically relevant even when they share few exact words with the question, improving recall on paraphrased or indirect queries.
Sentence-level chunking (3 sentences per chunk) balances granularity and context: single sentences are often too short to carry enough meaning, while full paragraphs can dilute the similarity signal.
The 0.2 similarity threshold filters out semantically unrelated chunks, keeping the context window focused and reducing noise passed to the LLM.
LLaMA 3.1 (8B) via Groq was chosen for fast, free inference during development.
Caching is implemented using a JSON file to store previously retrieved Wikipedia search results and page contents. This reduces redundant API calls and improves performance for repeated queries.
-----------------
Algorithm:
1. Load the GROQ_API_KEY from the environment using dotenv.
2. Load cached Wikipedia search results and page contents from a JSON file if it exists.
3. Prompt the user to enter a question.
4. Call the LLM to extract 3-5 keywords from the question.
5. Check if the query exists in the cache:
   - If yes, use cached search results/page content.
   - If no, search Wikipedia and store the results in the cache.
6. Split the article text into chunks of 3 sentences using regex sentence boundary detection.
7. Encode all chunks and the question into dense vectors using the sentence embedding model.
    - Clean up the article text by removing section headers and excess whitespace before chunking to improve embedding quality.
8. Compute cosine similarity between the question embedding and each chunk embedding.
9. Filter out chunks below the similarity threshold (0.2); sort remaining chunks in descending order.
10. Return the top 3 chunks as evidence passages.
11. Format a prompt containing the question and the top 3 chunks and send it to the LLM.
12. Print the generated answer followed by the numbered evidence passages (1-3).
"""
