#!pip install twitterscraper
#!pip install twint
#!pip install textblob
#!pip install tweepy

import os
import tweepy

# se connecter à l'API twitter

consumerKey = "X9D04kfQnEpivgxhfENOzoyFI"
consumerSecret = "UIe7omKcu6Q9pEp9vPlMtNvN2W73iO4JwwSg0KbmI5hB2tSn7h"
accessToken = "1392928077798326275-R2XoZd6N8ZSSD2f4XzRKRTGNXDKq6Y"
accessTokenSecret = "VBzhF9FeHwOkC6Ex5P0Z63aXbwb3UdxCQBlQLB2m8DtUe"
authentification = tweepy.OAuthHandler(consumerKey, consumerSecret)
authentification.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

#  récupérer les données à partir de Twitter

tweets = tweepy.Cursor(api.search,
                   q = requete,
                   lang = "fr",
                   since='2018-01-15').items(1000)

all_tweets = [tweet.text for tweet in tweets]

