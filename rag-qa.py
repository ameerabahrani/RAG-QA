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

import os
import re

import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_keywords(question): #get keywords from question to use as query for wikipedia search
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Extract 3-5 search keywords from this question. Return only the keywords, nothing else.\nQuestion: {question}"
        }]
    )
    return response.choices[0].message.content.strip()

def best_wiki_title(question, titles):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"Given this question: '{question}'\nWhich of these Wikipedia article titles is most relevant? Return only the title, nothing else.\nTitles:\n" + "\n".join(titles)
        }]
    )
    return response.choices[0].message.content.strip()


     
def get_WikiPage(query, question):
    try:
        searchResults = wikipedia.search(query)
        if not searchResults:
            print("No results found.")
            return None

        # Try top 5 results safely
        for title in searchResults[:5]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                return page.content
            except wikipedia.exceptions.DisambiguationError:
                continue
            except wikipedia.exceptions.PageError:
                continue

        print("Could not find a valid page.")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def chunkText(text, chunk_size=3):
    sentences = re.split(r'(?<=[.!?]) +', text) # split text into sentences using regex that looks for punctuation followed by space
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]) # join sentences to form a chunk of text
        chunks.append(chunk)
    return chunks

def topK(question, chunks, k=3):
    vectorizer = TfidfVectorizer(stop_words='english') # initialize TF-IDF vectorizer with English stop words
    chunk_vectors = vectorizer.fit_transform(chunks) # learn the vocabulary and idf from the chunks and return the term-document matrix
    question_vector = vectorizer.transform([question]) #transform to convert the question into the same vector space as the chunks using the learned vocabulary and idf from the chunks
    
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten() # compute cosine similarity between the question vector and each chunk vector, resulting in a 1D array of similarity scores

    scored_chunks = list(zip(chunks, similarities)) # pair each chunk with its similarity score
    scored_chunks = [c for c in scored_chunks if c[1] > 0.08] # filter out chunks with zero similarity
    scored_chunks.sort(key=lambda x: x[1], reverse=True) # sort the chunks
    
    
    return [c[0] for c in scored_chunks[:k]] # return the top k chunks based on similarity scores


def generate_answer(question, topK_chunks):
    context = "\n\n".join(topK_chunks)
    prompt = prompt = f"""
You are a helpful assistant. Use the following retrieved passages to answer the question concisely in one sentence. If you cannot find the answer, say you don't know.
Question: {question}
Context: {context}
Answer:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[ {"role" : "user", "content": prompt}]
    )
    return response.choices[0].message.content



def main():
    question = input("Enter your question: ")
    text = get_WikiPage(question, question)
    if not text:
        print("Failed to retrieve wiki page.")
        return
    chunks = chunkText(text)
    topK_chunks = topK(question, chunks)
    #for i, chunk in enumerate(topK_chunks, start=1):
     #   print(f"{i}. {chunk}")

    answer = generate_answer(question, topK_chunks)
    print(f"Answer: {answer}")
    print("Evidence:")
    for i, chunk in enumerate(topK_chunks, start=1):
        print(f"{i}. {chunk}")


if __name__ == "__main__":
    main()