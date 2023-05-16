from transformers import (
    BertTokenizerFast,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
import torch

# https://medium.com/ilb-labs-publications/fine-tuning-bert-for-a-regression-task-is-a-description-enough-to-predict-a-propertys-list-price-cf97cd7cb98a
class BertRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_bert=False):
        
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1
        
        self.bert = \
                   BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))    
        
    def forward(self, input_ids, attention_masks):
        
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs

    
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

def load_nn(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam")
    return model 


def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader

def train(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        print(epoch)
        print("-----")
        best_loss = 1e10
        model.train()
        for step, batch in enumerate(train_dataloader): 
            print(step)  
            batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)           
            loss = loss_function(outputs.squeeze(), 
                             batch_labels.squeeze())
            print(f"Loss: {loss}")
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
                
    return model

def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss = []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
    return test_loss

def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = \
                                  tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks).view(1,-1).tolist()[0]
    return output

def train_model():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    dataset = load_data()
    

    
    tokenized_dataset = tokenize_data(tokenizer, dataset["product_info"])

    input_ids = np.array(tokenized_dataset["input_ids"])
    attention_mask = np.array(tokenized_dataset["attention_mask"])
    relevance_scores = np.array(dataset["relevance"])
    
    seed=1
    test_size=0.1
    
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        input_ids, relevance_scores, test_size=test_size, random_state=seed
    )
    train_masks, test_masks, _, _ = train_test_split(
        attention_mask, relevance_scores, test_size=test_size, random_state=seed
    )
    
    train_dataloader = create_dataloaders(train_inputs, train_masks, 
                                        train_labels, 32)
    test_dataloader = create_dataloaders(test_inputs, test_masks, 
                                        test_labels, 32)
    
    model = BertRegressor(drop_rate=0.2)
    
    device = torch.device("cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,
                      eps=1e-8)
    
    epochs = 20
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    loss_function = nn.MSELoss()
    
    model = train(model, optimizer, scheduler, loss_function, epochs, 
              train_dataloader, device, clip_value=2)
    
    
    
    # train_labels = price_scaler.transform(train_labels.reshape(-1, 1))
    # test_labels = price_scaler.transform(test_labels.reshape(-1, 1))
    
    
    # dataset["token_type_ids"] = tokenized_dataset["token_type_ids"]
    # dataset["input_ids"]  = tokenized_dataset["input_ids"]
    # dataset["attention_mask"] = tokenized_dataset["attention_mask"]
    
    # X = dataset.drop(["product_info", "relevance"], axis=1)
    # y = dataset.drop(["product_info", "token_type_ids", "input_ids", "attention_mask"], axis=1)
    # y = dataset.drop(["product_info", "token_type_ids"], axis=1)

    # X['input_ids'] = X['input_ids'].map(lambda x: np.array(x))
    # X['token_type_ids'] = X['token_type_ids'].map(lambda x: np.array(x))
    # X['attention_mask'] = X['attention_mask'].map(lambda x: np.array(x))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )  
    
    
    
    # X_train = np.array(X_train.values.tolist())
    # X_test = np.array(X_test.values.tolist())
    # y_train = np.array(y_train.values.tolist())
    # y_test = np.array(y_test.values.tolist())

    # # print(X_test)
    # print(X_test[0])
    # print(X_test[1])
    
    # model = load_nn(input_shape=(None, None, 3, 512))
    # model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)
    # y_pred = model.predict(X_test)
    
    # for test, pred in zip(y_test,y_pred):
    #     print(f"Prediction {pred}, Truth {test}")
    
    # # y_pred has shape [14814,3,1], we take 1st prediction (TODO: maybe reshape y_test or at least double check)
    # # reshape into [14814,1] because that is shape of y_test 
    # y_pred = y_pred[:,0,:]
    # y_pred = np.reshape(y_pred, (14814,1))
    
    # mse = mean_squared_error(y_test, y_pred)

    # rmse = math.sqrt(mse)
    
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    
    # TODO: cross-validation, seed for reproducibility
    
    # APPROACH
    # Tokenize the input query and product using BERT to generate fixed-length vector representations.
    # Concatenate the vector representations of the query and product to form a single input vector.
    # Feed the input vector into a feedforward neural network that has been defined using TensorFlow's tf.keras API.
    # Train the neural network on a training set of labeled data, using the fit method of the tf.keras.Model class.
    # Evaluate the performance of the trained model on a held-out test set, using the evaluate method of the tf.keras.Model class.
    # Use the trained model to make relevance score predictions for new queries and products.
    


if __name__ == "__main__":
    train_model()