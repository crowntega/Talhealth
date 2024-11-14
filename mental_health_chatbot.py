# Install necessary libraries if running locally
# %pip install streamlit nltk scikit-learn

import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Force download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_most_relevant_sentence(query, sentences):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform sentences
    all_text = [query] + sentences
    tfidf_matrix = vectorizer.fit_transform(all_text)
    # Calculate similarity between query and all sentences
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    # Get index of most similar sentence
    max_sim_idx = similarities.argmax()
    return sentences[max_sim_idx], similarities[0][max_sim_idx]

def chatbot(query, knowledge_base):
    # Preprocess query and split knowledge base into sentences
    processed_query = preprocess(query)
    sentences = nltk.sent_tokenize(knowledge_base)
    processed_sentences = [preprocess(sent) for sent in sentences]
    
    # Get most relevant sentence
    response, similarity = get_most_relevant_sentence(processed_query, processed_sentences)
    
    if similarity < 0.1:
        return "I'm sorry, I don't have enough information to answer that question properly. Could you rephrase or ask something else?"
    
    return response

def main():
    st.title("TalHealth - Mental Health Chatbot")
    
    # Example knowledge base
    knowledge_base = """
    Mental health includes our emotional, psychological, and social well-being. It affects how we think, feel, and act.
    Schizophrenia is a severe mental disorder that affects a person’s ability to think, feel, and behave clearly.
    Aripiprazole is an antipsychotic medication used to treat schizophrenia and other conditions.
    """
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Text input for user query
    user_input = st.text_input("Ask me anything about mental health:")
    
    if user_input:
        # Get response from chatbot
        response = chatbot(user_input, knowledge_base)
        
        # Add to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Miklal", response))
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"👤 You: {message}")
        else:
            st.write(f"🤖 Miklal: {message}")

if __name__ == "__main__":
    main()
