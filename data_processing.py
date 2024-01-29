import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Import functions for data preprocessing
import re
import emoji
import string
from string import punctuation
import nltk
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# pre trained model for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv('youtube_multi_videos_comments.csv')
data=data.drop(['Unnamed: 0','author','video_id','public'],axis=1)
data.dropna(subset=['text'], inplace=True)

def comment_text_processing(text):

    # Remove URL links
    text = re.sub(r'http\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove emojis from the text
    text = emoji.demojize(text)
     # convert text into lowercase
    text = text.lower()
    # Tokenization
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the words back into a string
    processed_text = ' '.join(words)

    return processed_text

# Apply preprocessing to the comment text
data.text = data.text.apply(lambda text: comment_text_processing(text))

# using pretrained model for sentiment analysis
SIA = SentimentIntensityAnalyzer()
data["Positive"] = [SIA.polarity_scores(i)["pos"] for i in data["text"]]
data["Negative"] = [SIA.polarity_scores(i)["neg"] for i in data["text"]]
data["Neutral"] = [SIA.polarity_scores(i)["neu"] for i in data["text"]]
data['Compound'] = [SIA.polarity_scores(i)["compound"] for i in data["text"]]
score = data["Compound"].values
sentiment = []
for i in score:
    if i >= 0.05 :
        sentiment.append('Positive')
    elif i <= -0.05 :
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
data["Sentiment"] = sentiment
data=data.drop(['Positive','Negative','Neutral','Compound'],axis=1)

plt.figure(figsize=(5.5,7.5))
plt.pie(data['Sentiment'].value_counts(),autopct="%1.1f%%",labels=data['Sentiment'].value_counts().index)
plt.legend(loc='best')
plt.show()

from wordcloud import WordCloud
all_words = ' '.join([text for text in data['text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# convert categorical data to numerical
label_encoder = LabelEncoder()
data['Sentiment'] = label_encoder.fit_transform(data['Sentiment'])
processed_data = {
    'text':data.text,
    'Sentiment':data['Sentiment']
}

processed_data = pd.DataFrame(processed_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    processed_data['text'],
    processed_data['Sentiment'],
    test_size=0.2,
    random_state=55
)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # can adjust max_features based on data size
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# model
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

classifier = LinearSVC()
classifier.fit(X_train_tfidf.toarray(), y_train)
y_predict = classifier.predict(X_test_tfidf)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy.round(3)*100}%")

# Display additional metrics
print('\nClassification Report:')
print(classification_report(y_test, y_predict))

# metrics display
from sklearn import metrics
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

def predict_sentiment(user_input, vectorizer, classifier):
    #
    text = comment_text_processing(user_input)
    # Vectorize the user input using the TF-IDF vectorizer
    user_input_tfidf = vectorizer.transform([text])

    # Predict sentiment using the trained classifier
    prediction = classifier.predict(user_input_tfidf)

    return prediction[0]

# Example Usage:
user_text = input("Enter the text: ")
predicted_sentiment = predict_sentiment(user_text, tfidf_vectorizer, classifier)
ps = {0:'Negative',1:'Neutral',2:'Positive'}
print(f"Predicted Sentiment: {ps[predicted_sentiment]}")



import joblib
# Save the TF-IDF vectorizer and classifier
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(classifier, 'sentiment_classifier.joblib')

# Load the TF-IDF vectorizer and classifier
loaded_tfidf_vectorizer = joblib.load('/tfidf_vectorizer.joblib')
loaded_classifier = joblib.load('sentiment_classifier.joblib')

# Example usage for prediction with the loaded model
user_text = "I really enjoyed the movie, it was fantastic!"
text = comment_text_processing(user_text)
user_input_tfidf = loaded_tfidf_vectorizer.transform([text])
predicted_sentiment = loaded_classifier.predict(user_input_tfidf)

print(f"Predicted Sentiment: {predicted_sentiment[0]}")