import tweepy
import json
import schedule
import time
import datetime
import os
import csv
from main import train_Model

def initiate_api():
    try: 
        with open('config.json', 'r') as f:
            config = json.load(f)        
        auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
        auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
        api = tweepy.API(auth)
        return api
    except:
        print("Problems with config.json")
        return None

def isEnglish(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def write_csv(tweet):
    file_tweets = open("trending_tweets/biden_tweets.csv", "a+")
    writer = csv.writer(file_tweets)
    writer.writerow(tweet)
    file_tweets.close()

class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if(not(status.text).startswith('RT')):
            # write_csv([str(status.id), str(status.created_at), status.user.screen_name,status.text])
            print( status.text + "  --------------  " )
        

def main():
    api = initiate_api()
    
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

    myStream.filter(track=['biden'])
        
if __name__ == "__main__":
    main()

