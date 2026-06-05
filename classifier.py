from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train
model.fit(train.data, train.target)

# Evaluate
predictions = model.predict(test.data)
print(classification_report(test.target, predictions, target_names=test.target_names))
