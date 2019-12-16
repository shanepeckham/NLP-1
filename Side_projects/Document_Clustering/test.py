
import pyLDAvis
import pyLDAvis.sklearn
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
docs_raw = newsgroups.data
print(len(docs_raw))

tf_vectorizer = TfidfVectorizer(
    min_df=.05,
    max_df=.8,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    max_features=1000,
    ngram_range=(1,3),
    lowercase=True,
    stop_words=stopwords.words('english'))

dtm_tf = tf_vectorizer.fit_transform(docs_raw)
print(dtm_tf.shape)

lda_tf = LatentDirichletAllocation(
    n_components=20, 
    max_iter=100, 
    learning_method='online',
    verbose=True,)
lda_tf.fit(dtm_tf)

pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)

