
import re
import pickle
import tweepy

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
    
# # Testing
# sample = ['Hey you rocks! Thanks for your help!']
# sample = vectorizer.vectorize(sample).toarray()
# print(classifier.predict(sample))

# 

from tweepy import OAuthHandler
consumer_key = 'Fuud0U6n26KuAElDuIpzbe8ma'
consumer_secret = 'm4VSGOtbjbWXW6qyaMIKiohWDiO9SuzK24jTaFyHvxec1RTsid'
access_token = '422519723-qJYKCaux4X8yQJasifaPh8k6Db4aVcB3lWBBoIvs'
access_secret = 'xCJ69v9nKQ42vC7LphI3Wk7bdRbGHgMpLleDL83RrkY2J'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token, access_secret)

args = ['facebook']
api = tweepy.API(auth,timeout=10)

list_tweets = list()
query = args[0]

if len(args) == 1:
    for status in tweepy.Cursor(
        api.search, q=query+' -filter:retweeets', lang='en', result_type='recent').items(100):
        list_tweets.append(status.text)



