from flask import Flask, render_template,redirect, request
import googleapiclient.discovery
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
# Import functions for data preprocessing
import re
import emoji
import string
from string import punctuation

app = Flask(__name__)


# Load the TF-IDF vectorizer and classifier
vectorizer = joblib.load('tfidf_vectorizer.joblib')
classifier = joblib.load('sentiment_classifier.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    analysis_type = request.form.get('analysis_type')

    if analysis_type == 'text':
        # Redirect to the page for text analysis
        return render_template('/text_analysis.html')
    elif analysis_type == 'url':
        # Redirect to the page for URL analysis
        return render_template('/video_analysis.html')


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

@app.route('/predict-text', methods=['POST'])
def predict_text():
    comment = request.form['text']
    processed_comment = comment_text_processing(comment)
    features = vectorizer.transform([processed_comment])
    prediction = classifier.predict(features)[0]
    sentiment = get_sentiment_label(prediction)
    return render_template('text_analysis_result.html', comment=comment, sentiment=sentiment)

@app.route('/predict-video', methods=['POST'])
def predict_video_comments():
    video_id = request.form['video']
    comments_data = get_youtube_video_comments(video_id)
    comments =[]
    sentiments =[]
    for comment in comments_data['text']:
        processed_comment = comment_text_processing(comment)
        features = vectorizer.transform([processed_comment])
        prediction = classifier.predict(features)[0]
        sentiment = get_sentiment_label(prediction)
        comments.append(comment)
        sentiments.append(sentiment)
     # Zip the data
    zipped_data = zip(comments, sentiments)
    return render_template('video_result.html', zipped_data =zipped_data)

def get_youtube_video_comments(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyBDJBeKx7_8cEMDARCYUcncUXTEMd7dr0g"
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()
    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['textDisplay']
        ])
    df = pd.DataFrame(comments, columns=['text'])
    return df

# def preprocess_comment(comment):
#     # Comment preprocessing code here
#     comment = comment.lower()
#     comment = comment.translate(str.maketrans('', '', string.punctuation))
#     comment = remove_stopwords(comment)
#     comment = stem_words(comment)
#     return comment

# def remove_stopwords(comment):
#     stopwords_english = set(stopwords.words('english'))
#     comment_tokens = comment.split()
#     comment = ' '.join([word for word in comment_tokens if word not in stopwords_english])
#     return comment

# def stem_words(comment):
#     stemmer = PorterStemmer()
#     comment_tokens = comment.split()
#     comment = ' '.join([stemmer.stem(word) for word in comment_tokens])
#     return comment

def get_sentiment_label(prediction):
    if prediction == 0:
        return 'negative '
    elif prediction == 1:
        return 'neutral '
    elif prediction == 2:
        return 'positive '
    else:
        return 'unknown'


if __name__ == '__main__':
    app.run(debug=True, port=5151)