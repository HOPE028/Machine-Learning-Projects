# Setup

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
all_tweets = pd.read_json("random_tweets.json", lines=True)

# Is famous?

median_retweet_count = all_tweets['retweet_count'].median()

all_tweets['is_viral'] = all_tweets.apply(lambda tweet: tweet['retweet_count'] >= median_retweet_count, axis=1)


# average size of word, hashtag count, link count, followers count, following count, length (how many words)

all_tweets['average_length_of_word'] = all_tweets.apply(lambda tweet: len(tweet['text']) / len(tweet['text'].split()), axis=1)
all_tweets['hashtag_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis=1)
all_tweets['link_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('http'), axis=1)
all_tweets['follower_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
all_tweets['text_length'] = all_tweets.apply(lambda tweet: len(tweet['text'].split()), axis=1)
all_tweets['favorite_count'] = all_tweets.apply(lambda tweet: tweet['favorite_count'], axis=1)
all_tweets['verified_account'] = all_tweets.apply(lambda tweet: tweet['user']['verified'], axis=1)
all_tweets['default_profile_image'] = all_tweets.apply(lambda tweet: tweet['user']['default_profile_image'], axis=1)
all_tweets['default_profile'] = all_tweets.apply(lambda tweet: tweet['user']['default_profile'], axis=1)

# Labels and data

labels = all_tweets['is_viral']
data = all_tweets[['average_length_of_word', 'hashtag_count', 'link_count', 'follower_count', 'friends_count', 'text_length', 'favorite_count', 'verified_account', 'default_profile_image']]
scaled_data = scale(data, axis=0)

# Training and resaults

train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(train_data, train_labels)
score = classifier.score(test_data, test_labels)
print(score)

# Graphing resaults

scores = []

for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    score = classifier.score(test_data, test_labels)
    scores.append(score)
    
plt.plot(range(1,200), scores)

plt.show()