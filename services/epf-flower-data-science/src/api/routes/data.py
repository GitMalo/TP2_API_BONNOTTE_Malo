from fastapi.responses import RedirectResponse
from src.services.data import *
from fastapi import APIRouter

router = APIRouter()

@router.get("/download-dataset")
def download_dataset():
    return download_dataset_kaggle()

@router.get("/load-dataset")
def load_dataset():
    return load_dataset_kaggle()

@router.get("/preprocessing-data")
def preprocessing_data():
    return preprocessing_data_kaggle()

@router.get("/split-train-test")
def split_train_test():
    return split_train_test_kaggle()

@router.get("/train_model")
def training_model():
    return train_model

@router.get("/predict_model")
def prediction_model():
    return predict_model()

@router.get("/retrieve_firestore")
def firestore():
    return retrieve_firestore()

@router.get("/update-firestore")
def upd_firestore():
    return update_firestore()