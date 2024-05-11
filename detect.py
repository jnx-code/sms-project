import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load data from CSV file
data = pd.read_csv('spam_detection.csv', encoding='latin1')

# Select relevant columns
X = data['v2']
y = data['v1']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline for training
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the testing set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(pipeline, 'spam_detection_model.pkl')

# Example usage:
# Load the model
model = joblib.load('spam_detection_model.pkl')

# Predict whether a message is spam or not
def detect_spam(message):
    prediction = model.predict([message])
    if prediction[0] == 'spam':
        return "This message is spam."
    else:
        return "This message is not spam."

# Example usage:
message = "Congratulations! You've won a free vacation. Click here to claim."
print(detect_spam(message))