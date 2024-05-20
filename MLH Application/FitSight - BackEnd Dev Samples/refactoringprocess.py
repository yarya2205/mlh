import pandas as pd
import gensim
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


nltk.download('stopwords')
nltk.download('punkt')


df = pd.read_csv('/mnt/data/sentiment_reviews_withcount.csv')
reviews = df['review texts'].tolist()  


def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = [[word for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]
             for document in texts]
    return preprocessed_texts

preprocessed_reviews = preprocess_text(reviews)


dictionary = corpora.Dictionary(preprocessed_reviews)
corpus = [dictionary.doc2bow(text) for text in preprocessed_reviews]


num_topics = 3  
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=15, iterations=100)


def get_categories(ldamodel, num_topics):
    categories = []
    for i in range(num_topics):
        topic_terms = ldamodel.show_topic(i)
        category_keywords = ", ".join([word for word, _ in topic_terms])
        categories.append(category_keywords)
    return categories

categories = get_categories(lda_model, num_topics)

print("Generated Categories:")
for idx, category in enumerate(categories):
    print(f"Category {idx + 1}: {category}") 
