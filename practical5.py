# ----------------------------------------------------------------------------------
# PRACTICAL 5 : SENTIMENT ANALYSIS MODEL 
# ----------------------------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(accuracy_score,classification_report,confusion_matrix)

# ----------------------------------------------------------------------------------
# STEP 1: Create Dataset
# ----------------------------------------------------------------------------------

data={
    "text":[
        "I love this movie",
        "This movie is excellent",
        "The movie was amazing",
        "I really enjoyed the movie",
        "This is a fantastic product",
        "I am very happy with the service",
        "The product is wonderful",
        "The service was excellent",

        "I hate this movie",
        "This movie is terrible",
        "The movie was boring",
        "I did not enjoy the movie",
        "This is a bad product",
        "I am very unhappy with the service",
        "The product is horrible",
        "The service was disappointing"

    ],

    "sentiment":[
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",

        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# ----------------------------------------------------------------------------------
# STEP 2: Separate the Input and Output
# ----------------------------------------------------------------------------------

X= df["text"]
y= df["sentiment"]

# ----------------------------------------------------------------------------------
# STEP 3: Split Dataset
# ----------------------------------------------------------------------------------

X_train,X_test,y_train,y_test= train_test_split( X , y , test_size=0.2 , random_state=42 , stratify=y)

# ----------------------------------------------------------------------------------
# STEP 4: TF-IDF Vectorization
# ----------------------------------------------------------------------------------

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ----------------------------------------------------------------------------------
# STEP 5: Create Model
# ----------------------------------------------------------------------------------
 
model= LogisticRegression()

#----------------------------------------------------------------------------------
# STEP 6: Train Model
# ----------------------------------------------------------------------------------

model.fit(X_train_tfidf,y_train)

#----------------------------------------------------------------------------------
# STEP 7: Predict the model
# ----------------------------------------------------------------------------------

y_pred= model.predict(X_test_tfidf)

#----------------------------------------------------------------------------------
# STEP 8: Check Accuracy 
# ----------------------------------------------------------------------------------

accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#----------------------------------------------------------------------------------
# STEP 9: Make Classification Report
# ----------------------------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#----------------------------------------------------------------------------------
# STEP 10: Confusion Matrix
# ----------------------------------------------------------------------------------

print("\nConfusion Matrix:")
cm= confusion_matrix(
    y_test, y_pred,labels=["negative","positive"]
)

print(cm)

#----------------------------------------------------------------------------------
# STEP 11: Test New Text
# ----------------------------------------------------------------------------------

new_text=[
    "I really love this fantastic movie"
]

new_text_tfidf = tfidf.transform(new_text)

prediction = model.predict(new_text_tfidf)

print("\nNew Text:", new_text[0])

print("Predicted Sentiment:", prediction[0])
