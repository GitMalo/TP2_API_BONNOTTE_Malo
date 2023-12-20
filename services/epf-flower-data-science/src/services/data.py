import kaggle.api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os
import json
import joblib

def download_dataset_kaggle():
    """download the dataset iris and saves it in the directory save_dir
    input : nothing
    output : Dataset downloaded and saved successfully."""
    kaggle.api.authenticate()
    dataset_name = "uciml/iris"
    save_dir = "src/data"
    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True)

    return {"Dataset downloaded and saved successfully."}

def load_dataset_kaggle():
    """transform the dataset to a json and return it
    input : nothing
    output : json containing iris dataset"""
    df = pd.read_csv("src/data/Iris.csv")
    return df.to_json(orient="records")

def preprocessing_data_kaggle():
    """delete the 5 firest letters of the column Species and return the dataset in json
    input : nothing
    output : json containing iris dataset preprocessed"""
    df = pd.read_csv("src/data/Iris.csv")
    df = df.drop("Id", axis=1)
    df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    df = df.rename(columns={"Species": "target"})
    return df.to_json(orient="records")

def split_train_test_kaggle():
    """train test split the preprocessed dataset and return them in json
    input : nothing
    output : 2 json containing the training and test set"""  
    df = pd.read_json(preprocessing_data_kaggle())
    x_train, x_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.2)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    train_json = train.to_json(orient="records")
    test_json = test.to_json(orient="records")
    train.to_csv("src/data/train.csv", index=False)
    test.to_csv("src/data/test.csv", index=False)
    return train_json, test_json

def train_model():
    """train the model, saves it and saves the parameters of the model
    input : nothing
    output : Model trained"""
    json_split,_ = split_train_test_kaggle()
    data_train = pd.read_json(json_split)
    X_train = data_train.drop("target", axis=1)
    y_train = data_train["target"]

    model = SVC()
    model.fit(X_train, y_train)
    
    params = model.get_params()
    params_path = os.path.join("src/config/", "model_parameters.json")
    with open(params_path, "w") as f:
        json.dump(params, f)

    os.makedirs("src/models/", exist_ok=True)
    model_path = os.path.join("src/models/model.joblib")
    joblib.dump(model, model_path)  

    return {"model trained"}, model

def predict_model():
    model = train_model()[1]
    data_test = pd.read_json(split_train_test_kaggle()[1])
    X_test = data_test.drop("target", axis=1)
    y_pred = model.predict(X_test)
    return pd.DataFrame(y_pred).to_json(orient="records")