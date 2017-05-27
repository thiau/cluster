
# coding: utf-8

# In[139]:

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd


# In[140]:

nltk.data.path.append("nltk_data")
stemmer = SnowballStemmer('portuguese')
stopwords = nltk.corpus.stopwords.words('portuguese')


# In[141]:

df = pd.read_csv('datasets/documents.csv')
docs = df['docs'].tolist()


# In[142]:

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[143]:

def cluster(documents, num_clusters):
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0,min_df=0.0, stop_words=stopwords,
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    matrix = tfidf_vectorizer.fit_transform(documents)
    km = KMeans(n_clusters=num_clusters,init='random')
    km.fit_predict(matrix)
    clusters=km.labels_.tolist()
    
    result = {'question': documents, 'cluster_id': clusters}
    return pd.DataFrame(result, index=clusters).sort_values(by='cluster_id',ascending=True)


# In[144]:

cluster(documents=docs,num_clusters=5)


# In[ ]:



