import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Setting up tweets

new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)

new_york = new_york_tweets['text'].tolist()
london = london_tweets['text'].tolist()
paris = paris_tweets['text'].tolist()


# Setting up labels and data

data = new_york + london + paris
labels = [0] * len(new_york) + [1] * len(london) + [2] * len(paris)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1, test_size=0.2)


# Count Vectorizer

counter = CountVectorizer()
counter.fit(train_data)
train_count = counter.transform(train_data)
test_count = counter.transform(test_data)

# Fitting them and predicting them

classifier = MultinomialNB()
classifier.fit(train_count, train_labels)
prediction = classifier.predict(test_count)

# Accuracy 

print(accuracy_score(prediction, test_labels))