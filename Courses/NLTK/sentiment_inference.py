
import pickle

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
    
# Testing
sample = ['Hey you rocks! Thanks for your help!']
sample = vectorizer.vectorize(sample).toarray()
print(classifier.predict(sample))
