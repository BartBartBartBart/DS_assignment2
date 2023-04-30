from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    BertTokenizerFast
)
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf

LABEL_LIST = [
    1,
    1.25,
    1.33,
    1.5,
    1.67,
    1.75,
    2,
    2.25,
    2.33,
    2.5,
    2.67,
    2.75,
    3
]

LABEL2ID = {
    1: 0,
    1.25: 1,
    1.33: 2,
    1.5: 3,
    1.67: 4,
    1.75: 5,
    2: 6,
    2.25: 7,
    2.33: 8,
    2.5: 9,
    2.67: 10,
    2.75: 11,
    3: 12
}
 
ID2LABEL = {
    0: 1,
    1: 1.25,
    2: 1.33,
    3: 1.5,
    4: 1.67,
    5: 1.75,
    6: 2,
    7: 2.25,
    8: 2.33,
    9: 2.5,
    10: 2.67,
    11: 2.75,
    12: 3
}   


def compute_metrics(eval_pred):
    # predictions, labels = eval_pred
    pred_logits, labels = eval_pred

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [LABEL_LIST[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [LABEL_LIST[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    
    rmse = mean_squared_error(true_labels, predictions, squared=False)
    return {"rmse": rmse}
    
def load_data():
    # attr_df = pd.read_csv("./data/attributes.csv",encoding="utf-8")
    # submission_df = pd.read_csv("./data/sample_submission.csv",encoding="utf-8"
    # test_df = pd.read_csv("./data/test.csv",encoding="latin-1",on_bad_lines = 'warn')
    train_df = pd.read_csv("./data/train.csv",encoding="latin-1",on_bad_lines = 'warn')
    product_df = pd.read_csv("./data/product_descriptions.csv",encoding="utf-8")
    df = pd.merge(train_df, product_df, how='left', on='product_uid')
    df['product_info'] = df['search_term']+"\t"+df['product_title']+"\t"+df['product_description']
    df = df.drop(['id', 'product_uid', 'product_title', 'search_term','product_description'],axis=1)
    # df.replace({"relevance": LABEL2ID}, inplace=True)
    # print(df.head(5))
    return df

# 2.33
# ID2LABEL[2.33]

# input
# "query \tab product title product description"
# relevance label
# 2.33

def load_perceptron(input_shape=(768,)):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss="rmse", optimizer="adam")
    return model 

def train():
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                num_labels=7)
    args = TrainingArguments(
        "test-ner",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_steps=10,
        load_best_model_at_end=True,
        logging_steps=10,
    )


    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    perceptron = load_perceptron()
    
    
    # APPROACH
    # Tokenize the input query and product using BERT to generate fixed-length vector representations.
    # Concatenate the vector representations of the query and product to form a single input vector.
    # Feed the input vector into a feedforward neural network that has been defined using TensorFlow's tf.keras API.
    # Train the neural network on a training set of labeled data, using the fit method of the tf.keras.Model class.
    # Evaluate the performance of the trained model on a held-out test set, using the evaluate method of the tf.keras.Model class.
    # Use the trained model to make relevance score predictions for new queries and products.
    
    # trainer.train()


if __name__ == "__main__":
    # train()
    print(load_data().head(5))