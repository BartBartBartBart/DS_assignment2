from transformers import (
    BertTokenizerFast
)
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
# import torch
    
def load_data():
    train_df = pd.read_csv("./data/train.csv",encoding="latin-1",on_bad_lines = 'warn')
    product_df = pd.read_csv("./data/product_descriptions.csv",encoding="utf-8")
    df = pd.merge(train_df, product_df, how='left', on='product_uid')
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df = df.drop(['id', 'product_uid', 'product_title', 'search_term','product_description'],axis=1)
    return df


def tokenize_data(tokenizer, data):
    list_data = data.tolist()
    tokenized_data = tokenizer(list_data, truncation=True, padding="max_length")
    
    return {
        "input_ids": tokenized_data["input_ids"],
        "token_type_ids": tokenized_data["token_type_ids"],
        "attention_mask": tokenized_data["attention_mask"]
    }

def load_perceptron(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
    return model 

def train():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    dataset = load_data()
    
    tokenized_dataset = tokenize_data(tokenizer, dataset["product_info"])
        
    dataset["token_type_ids"] = tokenized_dataset["token_type_ids"]
    dataset["input_ids"]  = tokenized_dataset["input_ids"]
    dataset["attention_mask"] = tokenized_dataset["attention_mask"]
    
    X = dataset.drop(["product_info", "relevance"], axis=1)
    y = dataset.drop(["product_info", "token_type_ids", "input_ids", "attention_mask"], axis=1)

    X['input_ids'] = X['input_ids'].map(lambda x: np.array(x))
    X['token_type_ids'] = X['token_type_ids'].map(lambda x: np.array(x))
    X['attention_mask'] = X['attention_mask'].map(lambda x: np.array(x))

    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42
    )  
    
    # print(type(X_train["input_ids"].iloc[0]), type(X_test), type(y_train), type(y_test.iloc[0]))
    # X_train = X_train.map(lambda x: np.array(x))

    # # convert to list
    # X_train = X_train.values.tolist()
    # X_test = X_test.values.tolist()
    # y_train = y_train.values.tolist()
    # y_test = y_test.values.tolist()
       
    X_train = np.array(X_train.values.tolist())
    X_test = np.array(X_test.values.tolist())
    y_train = np.array(y_train.values.tolist())
    y_test = np.array(y_test.values.tolist())

       
    # print(X_train.iloc[0])
    
    perceptron = load_perceptron(input_shape=(None, None, 3, 512))
    perceptron.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    _, accuracy = perceptron.evaluate(X_test, y_test)
    y_pred = perceptron.predict(X_test)
    print("ACCURACY ", accuracy)
    
    mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)

    print(f"MSE: {mse}")
    
    # APPROACH
    # Tokenize the input query and product using BERT to generate fixed-length vector representations.
    # Concatenate the vector representations of the query and product to form a single input vector.
    # Feed the input vector into a feedforward neural network that has been defined using TensorFlow's tf.keras API.
    # Train the neural network on a training set of labeled data, using the fit method of the tf.keras.Model class.
    # Evaluate the performance of the trained model on a held-out test set, using the evaluate method of the tf.keras.Model class.
    # Use the trained model to make relevance score predictions for new queries and products.
    


if __name__ == "__main__":
    train()