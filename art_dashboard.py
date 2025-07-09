import streamlit as st
import praw
import pandas as pd
import re
from collections import Counter
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

spacy.cli.download("en_core_web_sm")

# Set up Streamlit page
st.set_page_config(page_title="Reddit ArtSensei Insights", layout="wide")
st.title("ArtSensei Reddit Insights Dashboard")

# Sidebar inputs
st.sidebar.header("Reddit API Credentials")
client_id = st.sidebar.text_input("Client ID", type="password")
client_secret = st.sidebar.text_input("Client Secret", type="password")
user_agent = "ArtSenseiDashboard"

subreddits = ['learnart', 'learntodraw', 'drawing']
limit = st.sidebar.slider("Number of Posts per Subreddit", 50, 1000, 200)

# Load NLP model
nlp = spacy.load('en_core_web_sm')

if client_id and client_secret:
    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

    # Fetch posts
    posts = []
    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).top(limit=limit):
            content = post.title + ' ' + post.selftext
            posts.append(content.lower())

    df = pd.DataFrame(posts, columns=['content'])

    # Clean text
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    df['cleaned'] = df['content'].apply(clean_text)
    all_text = ' '.join(df['cleaned'].tolist())

    # NLP Analysis
    doc = nlp(all_text)

    noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']

    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
    X = vectorizer.fit_transform(df['cleaned'])
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    noun_df = pd.DataFrame(Counter(noun_phrases).most_common(20), columns=['Noun Phrase', 'Frequency'])
    verb_df = pd.DataFrame(Counter(verbs).most_common(20), columns=['Verb', 'Frequency'])
    phrase_df = pd.DataFrame(words_freq[:20], columns=['Phrase', 'Frequency'])

    # WordCloud
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(all_text)
    st.subheader("Overall Word Cloud")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    st.subheader("Top Noun Phrases (What people talk about)")
    st.dataframe(noun_df)

    st.subheader("Top Verbs (Actions and Needs)")
    st.dataframe(verb_df)

    st.subheader("Top 2-3 Word Phrases")
    st.dataframe(phrase_df)

else:
    st.warning("Please enter your Reddit API credentials in the sidebar to begin analysis.")
