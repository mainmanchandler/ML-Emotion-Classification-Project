# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pandas as pd
import numpy as np

# %%
#df = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/train_emotion_dataset.csv")
#dftest = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/test_emotion_dataset.csv")
df = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/combined_samples_gt.csv")
#dftest = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/processed_test_sample_gt.csv")
df.head()

# %%
df['Ground Truth'].value_counts()

# %%
Xfeatures = df['Tweet']
ylabels = df['Ground Truth']
cv = CountVectorizer()
X = cv.fit_transform((Xfeatures.values.astype('U')).ravel())
cv.get_feature_names_out()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3, random_state=42)

#pipe_lr = Pipeline(steps=[("cv", CountVectorizer()), ("lr", LogisticRegression)])
#pipe_lr.fit(X_train, y_train)
nv_model = MultinomialNB()
nv_model.fit(X_train, y_train)
nv_model.score(X_test, y_test)


# %%
y_pred_for_nv = nv_model.predict(X_test)
y_pred_for_nv

# %%
test_text = ["I love love love this project this so much hehehehehhe"]
test_vector = cv.transform(test_text).toarray()

# %%
#make prediction:
nv_model.predict(test_vector)

# %%
# checking prediction probability, percentage/confidence score:
nv_model.predict_proba(test_vector)


# %%
#get all classes for the model
nv_model.classes_

# %%
np.max(nv_model.predict_proba(test_vector)) #max value percentage

# %%
#predict emotion function
def predicting_emotion_of_tweet(tweet, model):
    vector = cv.transform(tweet).toarray()
    prediction_of_model = model.predict(vector)
    prediction_probability = model.predict_proba(vector)
    prediction_percentage_for_every_tweet = dict(zip(model.classes_, prediction_probability[0]))
    print("Prediction: {}, Prediction_score: {}".format(prediction_of_model[0], np.max(prediction_probability)))
    return prediction_percentage_for_every_tweet

# %%
predicting_emotion_of_tweet(test_text, nv_model)


# %%
predicting_emotion_of_tweet(["I really dont like to run, in fact I hate hate hate running."], nv_model)

# %%
## Model Evaluating:
print(classification_report(y_test, y_pred_for_nv))

# %%
# Confusion Matrix
confusion_matrix(y_test, y_pred_for_nv)

#plot confusion matrix
#ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_for_nv), ylabels)

# %%
# make a predictions column and check what that is against the groud truth

df = df.dropna()
df['Predictions'] = ''
for i, row in df.iterrows():
    text = [row['Tweet']]
    vector = cv.transform(text).toarray()
    df.at[i, 'Predictions'] = nv_model.predict(vector)
    
df.head(30)

# %%
df['Match'] = ''

for i, row in df.iterrows():
    if (row['Predictions'] == row['Ground Truth']):
        df.at[i, 'Match'] = True
    else:
        df.at[i, 'Match'] = False   

df.head(20)

# %%
df['Match'].value_counts()


# %%
# count correct and incorrect
result = df['Match'].value_counts()

# find correct count
correct = result.iloc[0]

# find incorrect count
incorrect = result.iloc[1]

# find total
total = correct + incorrect

# find accuracy
accuracy = correct / total

# print accuracy
accuracy

# %%
# Export the dataset
# df.to_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/interim/multinomialNB_model_training.csv", index = False)


# %%
# Get the original dataset and run the model prediction on the Tweets (non-samples)
original_df = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/original_dataset.csv", encoding = "ISO-8859-1")

original_df = original_df.dropna()
original_df['Predictions'] = ''
original_df.head(30)

# %%
# insert model predictions
for i, row in original_df.iterrows():
    text = [row['Tweet']]
    vector = cv.transform(text).toarray()
    original_df.at[i, 'Predictions'] = nv_model.predict(vector)

original_df.head(30)


