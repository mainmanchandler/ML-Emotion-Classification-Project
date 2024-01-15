# imports
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split

"""
Description:   
    (USED FOR CODE TESTING)
    This function preprocesses the test_emotion_set & train_emotion_set that is used to train the emotion classification models

Parameters:
    input_filepath: file path of datasets to process (str)
    output_filepath_train: file path to save training csv (str)
    output_filepath_test: file path to save test csv (str)

Returns:
    Saves processed train_emotion_dataset.csv & test_emotion_dataset.csv to processed folder (.csv)
"""
def pre_process_emotion_training(input_filepath, output_filepath_train, output_filepath_test):

    # import data
    dataset_testing = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/test_emotion_set.txt", delimiter=';', header=None, names=['Text', 'Emotion'])
    dataset_training = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/train_emotion_set.txt", delimiter=';', header=None, names=['Text', 'Emotion'])
    dataset_validation = pd.read_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/train_emotion_set.txt", delimiter=';', header=None, names=['Text', 'Emotion'])

    #dataset_testing_small.head(20)
    #dataset_training_small.head()

    df = pd.concat([dataset_testing, dataset_training, dataset_validation])
    #df.head(20)

    ##convert to text objects to string (float issue), lowercase, replace ` with ' (text uses ` causes issues with keyword extraction) 
    df = df.dropna()
    df['Text'] = df['Text'].astype(str)
    df['Text'] = df['Text'].apply(str.lower)
    df['Text'] = df['Text'].str.replace("`", "'")

    ##remove stopwords
    stopwords_to_remove = stopwords.words("english")
    #print(stopwords_to_remove)
    df['Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_to_remove)]))

    ##remove special char, links, userhandles
    def remove_links_special_chars(text):
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
        return text
    df['Text'] = df['Text'].apply(remove_links_special_chars)
    #print("after special char, link, userhandle extraction: ")

    ##remove punctuation
    df['Text'] = df['Text'].str.replace(r'[^\w\s]+', '')
    #print("after punctuation extraction")
    df.head(20)

    #remove numerical values
    def remove_numerical_values(text):
        text = re.sub('[0-9]+', "", text)
        return text
    df['Text'] = df['Text'].apply(remove_numerical_values)

    ##### Split the data into training and testing #####
    train, test = train_test_split(df, test_size=0.2)

    ##### Save df to processed data folder as CSV #####
    train.to_csv(output_filepath_train, index = False)
    test.to_csv(output_filepath_test, index = False)


#change your filepath here and run the function to process the raw emotion training and testing datasets
input_filepath = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/"
output_filepath_train_sample = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/train_emotion_dataset.csv"
output_filepath_test_sample = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/test_emotion_dataset.csv"
pre_process_emotion_training(input_filepath, output_filepath_train_sample, output_filepath_test_sample) 