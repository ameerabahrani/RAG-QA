#Summary:
    # 1. Retrieval 
        # Represent docu/passages using TF-IDF 
        # Retrieve top-k passages that are most relevant to the input question
        #justify choice of TF-IDF
    #2. Answer Generation
        # Using retrieved passages, produce final answer using LLM of choice
    #3. Answer generated + Evidence
        #include top 1-3 passages that justify/support the answer generated

#System Design: 
    #1. Retrieval 
        # Wikipedia as the knowledge source
        # convert passages into vectors using TF-IDF
        #user question is also converted into vector using same TF-IDF model
        #compute cosine similarity between question vector and passage vectors to retrieve top-k relevant passages
        # return topK passages for justification
    #2. Answer Generation
        # Use a pre-trained LLM (e.g., GPT-3.5) to generate answer based on retrieved passages
        # Input to LLM: question + retrieved passages
        # return final answer
    #3. Evidence
        #print top 1-3 passages with numbering

#Decomposed: 
    # get wiki page 
    # split text into chunks
    #rank chunks using TF-IDF 
    #vectorize user query, cosine similarity, rank chunks
    #send to LLM ( q + chunk = ans)
    #return ans + chunks

import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_WikiPage(query):
    try:
        page = wikipedia.page(query)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error: {e}")
        return None
    except wikipedia.exceptions.PageError as e:
        print(f"Page error: {e}")
        return None
    
def chunkText(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def topK(question, chunks, k=3):
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks) # learn the vocabulary and idf from the chunks and return the term-document matrix
    question_vector = vectorizer.transform([question]) #transform to convert the question into the same vector space as the chunks using the learned vocabulary and idf from the chunks
    
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten() # compute cosine similarity between the question vector and each chunk vector, resulting in a 1D array of similarity scores
    topK_indices = similarities.argsort()[-k:][::-1] 
    
    return [chunks[i] for i in topK_indices]

def main():
    question = input("Enter your question: ")
    text = get_WikiPage(question)
    chunks = chunkText(text)
    topK_chunks = topK(question, chunks)
    for i, chunk in enumerate(topK_chunks, start=1):
        print(f"{i}. {chunk}")

if __name__ == "__main__":
    main()