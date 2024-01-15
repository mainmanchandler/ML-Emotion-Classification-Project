# imports
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split


"""
Description:   
    This function preprocesses the dataset used for statistic analysis and evaluation. Not the training and 
    testing sample sets for the models but the dataset that the best model is used on to make informed information for objectives 
    stated in abstract for report.

Parameters:
    input_filepath: file path of datasets to process (str)
    output_filepath_train: file path to save training csv (str)
    output_filepath_test: file path to save test csv (str)

Returns:
    Saves processed dataset.csv to processed folder (.csv)
"""
def preprocessing(input_filepath, output_filepath): #, output_filepath_train_sample, output_filepath_test_sample):

    # import data
    dataset_testing_small = pd.read_csv(input_filepath + "test.csv", encoding = "ISO-8859-1")
    dataset_training_small = pd.read_csv(input_filepath + "train.csv", encoding = "ISO-8859-1")


    ##### data1 #####

    ## rename columns by index
    d2_cols = dataset_testing_small.columns
    dataset_testing_small.rename(columns = {d2_cols[0]:'ID', d2_cols[1]:'Tweet', d2_cols[2]:'Sentiment', d2_cols[3]:'Time', d2_cols[4]:'Age'}, inplace=True)

    ## view df
    dataset_testing_small.head()


    ##### data2 #####

    ## drop unnecessary coolumns 
    dataset_training_small = dataset_training_small.drop(["selected_text"], axis = 1)

    ## rename columns by index
    d4_cols = dataset_training_small.columns
    dataset_training_small.rename(columns = {d4_cols[0]:'ID', d4_cols[1]:'Tweet', d4_cols[2]:'Sentiment', d4_cols[3]:'Time', d4_cols[4]:'Age'}, inplace=True)

    ## view df
    dataset_training_small.head()


    ##### merge dataframes #####
    frames = [dataset_testing_small, dataset_training_small]
    df = pd.concat(frames)
    df

    ##### Run Filtering on the dataframe #####

    ##convert to text objects to string (float issue), lowercase, replace ` with ' (text uses ` causes issues with keyword extraction) 
    df = df.dropna()
    df['Tweet'] = df['Tweet'].astype(str)
    df['Tweet'] = df['Tweet'].apply(str.lower)
    df['Tweet'] = df['Tweet'].str.replace("`", "'")


    ##remove stopwords
    stopwords_to_remove = stopwords.words("english")
    df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_to_remove)]))


    ##remove special char, links, userhandles
    def remove_links_special_chars(text):
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
        return text
    df['Tweet'] = df['Tweet'].apply(remove_links_special_chars)


    ##remove punctuation
    df['Tweet'] = df['Tweet'].str.replace(r'[^\w\s]+', '')

    #remove numerical values
    def remove_numerical_values(text):
        text = re.sub('[0-9]+', "", text)
        return text
    df['Tweet'] = df['Tweet'].apply(remove_numerical_values)

    ##### Save df to processed data folder as CSV #####
    df.to_csv(output_filepath, index = False)

    

#change your filepath here and run the function to process the raw dataset
input_filepath = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/"
output_filepath = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/original_dataset.csv"
#preprocessing(input_filepath, output_filepath) 


"""
Description:   
    This function creates two smaller sampling datasets (train_sample & test_sample) for human validation and ground truth testing.

Parameters:
    input_filepath: file path of datasets to process (str)
    output_filepath_train: file path to save train_sample csv (str)
    output_filepath_test: file path to save test_sample csv (str)

Returns:
    Saves processed train_sample.csv & test_sample.csv to raw folder (.csv)
"""
def create_human_review_sample_set(input_filepath, output_raw_filepath_train, output_raw_filepath_test):

    # import data
    dataset_testing_small = pd.read_csv(input_filepath + "test.csv", encoding = "ISO-8859-1")
    dataset_training_small = pd.read_csv(input_filepath + "train.csv", encoding = "ISO-8859-1")

    d2_cols = dataset_testing_small.columns
    dataset_testing_small.rename(columns = {d2_cols[0]:'ID', d2_cols[1]:'Tweet', d2_cols[2]:'Sentiment', d2_cols[3]:'Time', d2_cols[4]:'Age'}, inplace=True)

    ## view df
    dataset_testing_small.head()

    ## drop unnecessary coolumns 
    dataset_training_small = dataset_training_small.drop(["selected_text"], axis = 1)

    ## rename columns by index
    d4_cols = dataset_training_small.columns
    dataset_training_small.rename(columns = {d4_cols[0]:'ID', d4_cols[1]:'Tweet', d4_cols[2]:'Sentiment', d4_cols[3]:'Time', d4_cols[4]:'Age'}, inplace=True)

    ## view df
    dataset_training_small.head()

    frames2 = [dataset_testing_small, dataset_training_small]
    df = pd.concat(frames2)

    #df drop na
    df = df.dropna()

    #get sample from df
    df_sample = df.sample(n=1000)
    train_sample, test_sample = train_test_split(df_sample, test_size=0.2)

    #save sample to interim
    train_sample.to_csv(output_raw_filepath_train, index = False)
    test_sample.to_csv(output_raw_filepath_test, index = False)


#change your filepath here and run the function to create sample datasets
input_filepath = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/"
output_raw_filepath_train_sample = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/train_sample.csv"
output_raw_filepath_test_sample = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/test_sample.csv"
#create_human_review_sample_set(input_filepath, output_raw_filepath_train_sample, output_raw_filepath_test_sample) 



"""
Description:   
    This function preprocesses the human sampled datasets used for model evaluation and insight.

Parameters:
    input_filepath: file path of datasets to process (str)
    output_filepath_train: file path to save training csv (str)
    output_filepath_test: file path to save test csv (str)

Returns:
    Saves processed processed_test_sample_gt.csv & processed_train_sample_gt.csvto processed folder (.csv)
"""
def preprocess_samples(input_filepath, output_filepath_train_sample_processed, output_filepath_test_sample_processed):
    
    # import data
    dataset_testing_small = pd.read_csv(input_filepath + "test_sample.csv", encoding = "ISO-8859-1")
    dataset_training_small = pd.read_csv(input_filepath + "train_sample.csv", encoding = "ISO-8859-1")

    ## rename columns by index
    d2_cols = dataset_testing_small.columns
    dataset_testing_small.rename(columns = {d2_cols[0]:'ID', d2_cols[1]:'Tweet', d2_cols[2]:'Sentiment', d2_cols[3]:'Time', d2_cols[4]:'Age', d2_cols[9]:"Ground Truth"}, inplace=True)

    ## view df
    dataset_testing_small.head()

    ## rename columns by index
    d4_cols = dataset_training_small.columns
    dataset_training_small.rename(columns = {d4_cols[0]:'ID', d4_cols[1]:'Tweet', d4_cols[2]:'Sentiment', d4_cols[3]:'Time', d4_cols[4]:'Age'}, inplace=True)

    ## view df
    dataset_training_small.head()


    ##### merge dataframes #####
    df = dataset_training_small

    ##### Run Filtering on the dataframe #####

    ##convert to text objects to string (float issue), lowercase, replace ` with ' (text uses ` causes issues with keyword extraction) 
    df = df.dropna()
    df['Tweet'] = df['Tweet'].astype(str)
    df['Tweet'] = df['Tweet'].apply(str.lower)
    df['Tweet'] = df['Tweet'].str.replace("`", "'")

    ##remove stopwords
    stopwords_to_remove = stopwords.words("english")
    #print(stopwords_to_remove)
    df['Tweet'] = df['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_to_remove)]))


    ##remove special char, links, userhandles
    def remove_links_special_chars(text):
        text = re.sub('@[^\s]+', '', text)
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
        return text
    df['Tweet'] = df['Tweet'].apply(remove_links_special_chars)


    ##remove punctuation
    df['Tweet'] = df['Tweet'].str.replace(r'[^\w\s]+', '')

    #remove numerical values
    def remove_numerical_values(text):
        text = re.sub('[0-9]+', "", text)
        return text
    df['Tweet'] = df['Tweet'].apply(remove_numerical_values)

    ##### Split the data into training and testing #####
    df.to_csv("C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/combined_samples_gt.csv", index = False)

    #train, test = train_test_split(df, test_size=0.2)
    
    ##### Save df to processed data folder as CSV #####
    #train.to_csv(output_filepath_train_sample_processed, index = False)
    #test.to_csv(output_filepath_test_sample_processed, index = False)


#change your filepath here and run the function to process the raw sample datasets
input_filepath = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/raw/"
output_filepath_train_sample_processed = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/processed_trained_sample_gt.csv"
output_filepath_test_sample_processed = "C:/Users/chand/OneDrive/Documents/Bruh/final-project-190688910-190988840/data/processed/processed_test_sample_gt.csv"
preprocess_samples(input_filepath, output_filepath_train_sample_processed, output_filepath_test_sample_processed) 