"""
train.py
Implementation of the effect of a BERT tokenizer on the performance of 
a randomforest + baggingregressor on a relevance score prediction task.

Course: Data Science
Date: May 2023
Authors:
  - Lisanne Wallaard, s2865459, Data Science & Artificial Intelligence
  - Bart den Boef, s2829452, Data Science & Artificial Intelligence
"""

from transformers import (
    BertTokenizerFast
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
    
    
def str_stemmer(s, stemmer):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
    str1 = str(str1)
    str2 = str(str2)
    return sum(int(str2.find(word)>=0) for word in str1.split())


def load_data():
    train_df = pd.read_csv("./data/train.csv",encoding="latin-1",on_bad_lines = 'warn')
    product_df = pd.read_csv("./data/product_descriptions.csv",encoding="utf-8")
    df = pd.merge(train_df, product_df, how='left', on='product_uid')
    df['len_of_query'] = df['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    df = df.drop(['id'],axis=1)
    return df


def tokenize_data(tokenizer, data):
    list_data = data.tolist()
    tokenized_data = tokenizer(list_data, truncation=True, padding="longest")
    
    return {
        "input_ids": tokenized_data["input_ids"],
        "token_type_ids": tokenized_data["token_type_ids"],
        "attention_mask": tokenized_data["attention_mask"]
    }
    
    
# Generates X and y according to the baseline approach
# Uses the snowball stemmer
def dataset_baseline(dataset, stemmer):
    # Stemming with snowballstemmer
    dataset['search_term'] = dataset['search_term'].map(lambda x:str_stemmer(x,stemmer))
    dataset['product_title'] = dataset['product_title'].map(lambda x:str_stemmer(x,stemmer))
    dataset['product_description'] = dataset['product_description'].map(lambda x:str_stemmer(x,stemmer))
    # Count common words
    dataset['product_info'] = dataset['search_term']+"\t"+dataset['product_title']+"\t"+dataset['product_description']
    dataset['word_in_title'] = dataset['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    dataset['word_in_description'] = dataset['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    # Split in X and y
    dataset = dataset.drop(['search_term','product_title','product_description','product_info'],axis=1)
    X = dataset.drop(["relevance"], axis=1)
    y = dataset["relevance"]

    return X, y


# Generates X and y using the BERT tokenizer
def dataset_tokenizer(tokenizer, dataset):    
    # Tokenize the text in the dataset
    tokenized_query = tokenize_data(tokenizer, dataset["search_term"])  
    tokenized_description = tokenize_data(tokenizer, dataset["product_description"])    
    tokenized_title = tokenize_data(tokenizer, dataset["product_title"])  
    # Count common words
    dataset["word_in_title"] = str_common_word(tokenized_query["input_ids"], tokenized_title["input_ids"])
    dataset["word_in_description"] = str_common_word(tokenized_query["input_ids"], tokenized_description["input_ids"])
    # Split in X and Y
    dataset = dataset.drop(['search_term', 'product_title', 'product_description'],axis=1)
    X = dataset.drop(["relevance"], axis=1)
    y = dataset["relevance"]

    return X, y


# Generates X and y using the snowball stemmer
# Includes tokenized product info (tokenized using the BERT tokenizer)
def dataset_product_info(tokenizer, dataset, stemmer):   
    # Tokenize the text in the dataset
    tokenized_query = tokenize_data(tokenizer, dataset["search_term"])  
    tokenized_description = tokenize_data(tokenizer, dataset["product_description"])    
    tokenized_title = tokenize_data(tokenizer, dataset["product_title"])  
    
    # Stemming with snowballstemmer
    dataset['search_term'] = dataset['search_term'].map(lambda x:str_stemmer(x,stemmer))
    dataset['product_title'] = dataset['product_title'].map(lambda x:str_stemmer(x,stemmer))
    dataset['product_description'] = dataset['product_description'].map(lambda x:str_stemmer(x,stemmer))
    
    # Count common words
    dataset['product_info'] = dataset['search_term']+"\t"+dataset['product_title']+"\t"+dataset['product_description']
    dataset['word_in_title'] = dataset['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
    dataset['word_in_description'] = dataset['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
    
    # # Add tokenized text as feature to the dataset
    dataset["search_term"] = np.array(tokenized_query["input_ids"])
    dataset["product_description"] = np.array(tokenized_description["input_ids"])
    dataset["product_title"] = np.array(tokenized_title["input_ids"])
    
    # Split in X and y
    dataset = dataset.drop(['product_info'],axis=1)
    X = dataset.drop(["relevance"], axis=1)
    y = dataset["relevance"]

    return X, y


# Generates X and y using the BERT tokenizer, and includes the tokenized product info
def dataset_tokenizer_product_info(tokenizer, dataset):    
    # Tokenize the text in the dataset
    tokenized_query = tokenize_data(tokenizer, dataset["search_term"])  
    tokenized_description = tokenize_data(tokenizer, dataset["product_description"])    
    tokenized_title = tokenize_data(tokenizer, dataset["product_title"])  
    # Count common words
    dataset["word_in_title"] = str_common_word(tokenized_query["input_ids"], tokenized_title["input_ids"])
    dataset["word_in_description"] = str_common_word(tokenized_query["input_ids"], tokenized_description["input_ids"])
    # Add tokenized text as feature to the dataset
    dataset["search_term"] = np.array(tokenized_query["input_ids"])
    dataset["product_description"] = np.array(tokenized_description["input_ids"])
    dataset["product_title"] = np.array(tokenized_title["input_ids"])
    # Split in X and y
    X = dataset.drop(["relevance"], axis=1)
    y = dataset["relevance"]
    
    return X, y

# Plots the distribution of relevance scores in the dataset
def plot_relevance_score(relevance_score):
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=False)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(relevance_score, bins=7)
    axs[1].boxplot(relevance_score)

    # Set title
    fig.suptitle("Distribution of Relevance Scores", fontsize="x-large")

    # Set labels
    axs[0].set_xlabel("Relevance Score")
    axs[0].set_ylabel("Amount of ratings")

    axs[1].set_xlabel("")
    axs[1].set_ylabel("Relevance Scores")

    plt.show()
    
# Plots the importance of features in the model
def plot_importance_features(feature_names, X_train, y_train, seeds):    
    # Calculate the Importance of the features
    feature_importance = np.zeros(len(feature_names))
    for i in seeds:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=i)
        rf.fit(X_train, y_train)
        feature_importance = np.add(feature_importance, rf.feature_importances_)
    feature_importance = feature_importance/len(seeds)

    # Sort the features on Importance
    index_sorted = feature_importance.argsort()

    # Plot the Importance of the features
    plt.barh(feature_names[index_sorted], feature_importance[index_sorted])
    plt.xlabel("Feature Importance")
    plt.title("Features sorted by Importance")
    plt.show()
    
# Runs the experiment with seeds
def run_experiment(X_train, X_test, y_train, y_test, debug, seeds):
    # Initialize the average RMSE over 10 runs
    avg_rsme = 0
    for seed in seeds:
        rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=seed)
        clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=seed)  
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # For debugging purposes:
        #
        # for test, pred in zip(y_test,y_pred):
        #     print(f"Prediction {pred}, Truth {test}")
        
        # Calculate RMSE
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        
        avg_rsme += rmse
        
        # Print RMSE and MSE for each run
        if debug:
            print(f"MSE: {mse}")
            print(f"RMSE: {rmse}")
           
    # Print average RMSE over 10 runs
    print(f"Average RMSE over 10 runs: {avg_rsme/10}")
    
# Optimizes hyperparameters of the model
def hyperparameter_optimization(X_train, y_train):
    params = {
            'bootstrap': [True, False],
            'max_features': [0.2, 0.4, 0.6, 0.8, 1],
            'max_samples': [1,2,4,6,8,10],
            'n_estimators': [10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
            'oob_score': [True, False], 
            'verbose': [0,  5,10,15,20,25,30,35,40],  
            'warm_start': [True, False]
        }
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=0)
    rscv = RandomizedSearchCV(clf, params, random_state=0)
    search = rscv.fit(X_train, y_train)
    print(search.best_params_)
    

# Train model on specified version of dataset
def train(with_tokenizer=False, with_product_info=False, hp_optimization=False, debug=False, plot_data=False, plot_importance=False):
    
    # Load tokenizer and dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    stemmer = SnowballStemmer('english')
    dataset = load_data()
    
    # Set seeds for reproducibility
    seeds = [0,1,2,3,4,5,6,7,8,9]
    
    # Load specified version of dataset
    if with_tokenizer and with_product_info:
        print("Running BERT Tokenizer + Tokenized Product Info")
        X, y = dataset_tokenizer_product_info(tokenizer, dataset)
    elif with_product_info:
        print("Running Baseline + Tokenized Product Info")
        X, y = dataset_product_info(tokenizer, dataset, stemmer)
    elif with_tokenizer:
        print("Running BERT Tokenizer")
        X, y = dataset_tokenizer(tokenizer, dataset)
    else:
        X, y = dataset_baseline(dataset, stemmer)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=47
    )  
    
    # Hyperparameter optimization
    if hp_optimization:
        # Run hyperparameter optimization
        hyperparameter_optimization(X_train, y_train)
        
        # Optimal params:
        # bootstrap= True,
        # max_features= 0.8,
        # max_samples= 10,
        # n_estimators= 10,
        # oob_score= False,
        # verbose= 40,
        # warm_start= False, 
        
    else:
        # Run experiment with seeds
        run_experiment(X_train, X_test, y_train, y_test, debug, seeds)
        
    if plot_data:
        # Plot the distribution of relevance scores
        plot_relevance_score(y)
        
    if plot_importance:
        # Names of features in the model
        feature_names = X.columns
        # Plot the importance of the features
        plot_importance_features(feature_names, X_train, y_train, seeds)
    
    
if __name__=="__main__":
    # Turn 'with_tokenizer' and 'with_product_info' on or off to toggle between different setups
    # Set hp_optimization to True to do hyperparameter optimization
    train(
        with_tokenizer=False,
        with_product_info=True,
        hp_optimization=False,
        debug=False, 
        plot_data=False,
        plot_importance=False
    )
