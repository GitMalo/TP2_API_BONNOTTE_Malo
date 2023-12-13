import kaggle.api
import pandas as pd
from sklearn.model_selection import train_test_split

def download_dataset_kaggle():
    kaggle.api.authenticate()
    dataset_name = "uciml/iris"
    save_dir = "services/epf-flower-data-science/src/data"
    kaggle.api.dataset_download_files(dataset_name, path=save_dir, unzip=True)

    return {"Dataset downloaded and saved successfully."}

def load_dataset_kaggle():
    df = pd.read_csv("services/epf-flower-data-science/src/data/Iris.csv")
    return df.to_json(orient="records")

def preprocessing_data_kaggle():
    df = pd.read_csv("services/epf-flower-data-science/src/data/Iris.csv")
    df["Species"] =  df["Species"].apply(lambda x: x[5:])
    return df.to_json(orient="records")

def split_train_test_kaggle():
    df = pd.read_csv("services/epf-flower-data-science/src/data/Iris.csv")
    df["Species"] =  df["Species"].apply(lambda x: x[5:])
    x_train, x_test, y_train, y_test = train_test_split(df.drop("Species", axis=1), df["Species"], test_size=0.2)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    train_json = train.to_json(orient="records")
    test_json = test.to_json(orient="records")
    train.to_csv("services/epf-flower-data-science/src/data/train.csv", index=False)
    test.to_csv("services/epf-flower-data-science/src/data/test.csv", index=False)
    return train_json, test_json

    
