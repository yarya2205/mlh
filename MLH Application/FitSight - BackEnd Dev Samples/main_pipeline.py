import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
import pandas as pd

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertTokenizer as bt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertModel

import pandas as pd
import gensim
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re


nltk.download('punkt')
nltk.download('stopwords')



bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()


nltk.download('stopwords')
nltk.download('punkt')

df1 = pd.read_csv('sentiment_reviews_withcount.csv')
reviews = df1['review_text'].tolist() 

df = pd.DataFrame(reviews, columns=['Sentences'])


df['id'] = range(1, len(df) + 1)



def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = [
        [word for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]
        for document in texts]
    return preprocessed_texts


preprocessed_reviews = preprocess_text(reviews)


dictionary = corpora.Dictionary(preprocessed_reviews)
corpus = [dictionary.doc2bow(text) for text in preprocessed_reviews]


num_topics = 5  
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=15,
                            iterations=100)


additional_stopwords = {'get', 'great', 'like', 'really', 'good', 'gym', 'place', 'love', 'hate', 'one', 'trainer'}  # Add more words as needed



def preprocess_text(text):
    stop_words = stopwords.words('english')
    text = text.lower()  
    words = [word for word in text.split() if word not in stop_words]

    return " ".join(words)


def get_bert_embeddings(sentences):

    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = bert_model(**tokenized_sentences)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings

def calculate_center(df):
    centers = {}
    for cluster in df['Cluster'].unique():
        cluster_embeddings = df[df['Cluster'] == cluster]['bert_embeddings']
        center = cluster_embeddings.apply(pd.Series)
        center = center.mean()
        centers[cluster] = center.tolist()
    return centers


def find_closest_sentence(df, centers):
    closest_sentences = {}
    for cluster, center in centers.items():
        distances = [np.linalg.norm(np.pad(embedding, (0, len(center) - len(embedding)), 'constant') - center)
                        for
                        embedding in df[df['Cluster'] == cluster]['Vector'].values]
        closest_index = distances.index(min(distances))
        closest_sentences[cluster] = df[df['Cluster'] == cluster]['Sentences'].values[closest_index]
    return closest_sentences


def get_combined_categories(ldamodel, num_topics, num_keywords=5):

    all_keywords = []
    for i in range(num_topics):
        topic_terms = ldamodel.show_topic(i)
        all_keywords.extend([word for word, _ in topic_terms])


    keyword_counts = Counter(all_keywords)


    filtered_keywords = {word: count for word, count in keyword_counts.items() if word not in additional_stopwords}

    most_common_keywords = [word for word, count in Counter(filtered_keywords).most_common(num_keywords)]
    return most_common_keywords


#review data type -- sentiment empty
# -->

# Sample gym reviews
#reviews = [
    #"The staff at this gym are incredibly friendly and helpful. They always go the extra mile to make sure I have a great workout experience.",
    #"The equipment is top-notch and well-maintained. They have a wide variety of machines for all my training needs.",
   # "The gym is always clean and well-organized. It's a pleasure to work out in such a pleasant environment.",
  #  "The staff could be a bit more attentive, but the equipment is good overall.",
 #   "This gym is a bit dirty at times, but the staff is friendly and the classes are great.",
#]

df = pd.DataFrame(reviews, columns=['Sentences'])


labels = get_combined_categories(lda_model, num_topics)


processed_reviews = [preprocess_text(review) for review in reviews]
df['processed_sentences'] = processed_reviews


label_tracker_dict = {}

for i in range(0,len(labels)):
    label_tracker_dict[i] = labels[i]


label_texts = [" ".join([label, "review is"]) for label in labels]



tokenizer = bt.from_pretrained('bert-base-uncased')

embeddings = get_bert_embeddings(processed_reviews)
label_embeddings = get_bert_embeddings(label_texts)

df['bert_embeddings'] = embeddings.tolist()


# HDBSCAN clustering
clusterer = HDBSCAN(min_cluster_size=2, 
                             min_samples=2,         
                             metric='euclidean',
                             cluster_selection_method='leaf', 
                             prediction_data=True)
  
clusterer.fit(embeddings)


cluster_labels = clusterer.labels_

df['Cluster'] = cluster_labels

centers = calculate_center(df)


threshold = 0.62 
assigned_labels = {}
for cluster_id in centers.keys():
    centroid = centers[cluster_id]
    if cluster_id == -1:
        assigned_labels[cluster_id] = [-1]
        continue
    centroid = pd.Series(centroid)
    assigned_labels[cluster_id] = []
    for label_id, label_vector in enumerate(label_embeddings):
        label_vector = pd.Series(label_vector)
        similarity = cosine_similarity(centroid.values.reshape(1, -1), label_vector.values.reshape(1, -1))[0][0]
        if similarity > threshold:
            curr_labels = assigned_labels[cluster_id]
            curr_labels.append(label_id)
            assigned_labels[cluster_id] = curr_labels

for cluster_id in centers.keys():
    if (len(assigned_labels[cluster_id]) == 0):
        centroid = centers[cluster_id]
        if (assigned_labels[cluster_id] == [-1]):
            continue
        centroid = pd.Series(centroid)
        maximum = -1
        for label_id, label_vector in enumerate(label_embeddings):
            label_vector = pd.Series(label_vector)
            similarity = cosine_similarity(centroid.values.reshape(1, -1), label_vector.values.reshape(1, -1))[0][0]
            if similarity > maximum:
                assigned_labels[cluster_id] = [label_id]
                maximum = similarity



for i in range(0, len(processed_reviews)):
    if df['Cluster'][i] == -1:
        maximum = -1
        for label_id, label_vector in enumerate(label_embeddings):
            label_vector = pd.Series(label_vector)
            similarity = cosine_similarity(embeddings[i].reshape(1, -1), label_vector.values.reshape(1, -1))[0][0]
            if similarity > maximum:
                df['Cluster'][i] = label_id
                maximum = similarity



df['assigned_label'] = df['Cluster'].map(assigned_labels)
df['named_labels']  = df['assigned_label'].apply(lambda x: [label_tracker_dict[num] for num in x])
print('breakpoint')

for i in range(0, len(processed_reviews)):
    for j in labels:
        if j in processed_reviews[i]:
            if j in df['named_labels'][i]:
                continue
            else:
                curr = df['named_labels'][i]
                curr.append(j)

print("Reviews:")
for i, review in enumerate(reviews):
    print(f"- Review {i+1}: {review}")

print("\nClusters and assigned labels:")
for cluster_id, labels in assigned_labels.items():
    print(f"- Cluster {cluster_id+1}:", ", ".join(str(labels)))


from textblob import TextBlob
from collections import defaultdict
sentiments = []
polarities = []

for review in reviews:
    blob = TextBlob(review)
    sentiments.append(blob.sentiment[0] * 2.5 + 2.5)
    polarities.append(blob.sentiment[1]  * 2.5 + 2.5)

df['sentiment'] = sentiments
df['polarity'] = polarities
categories_summaries_sentiments = defaultdict(list)
categores_summaries_polarities = defaultdict(list)
for i in range(0, len(df)):
    for j in df['named_labels'][i]:
        categories_summaries_sentiments[j].append(df['sentiment'][i])
        categores_summaries_polarities[j].append(df['polarity'][i])

df_summaries = pd.DataFrame(columns=['Category', 'Average Sentiment', 'Average Polarity'])
for i in categories_summaries_sentiments.keys():
    
    df_summaries.loc[len(df_summaries)] = [i, sum(categories_summaries_sentiments[i])/len(categories_summaries_sentiments[i]), sum(categores_summaries_polarities[i])/len(categores_summaries_polarities[i])]
print(df_summaries)

print(df)

#6236.21