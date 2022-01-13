import os
from dotenv import load_dotenv
import re

import pymongo
from pymongo import MongoClient

import pandas as pd
import numpy as np


class Database:
    load_dotenv()
    _USER = os.getenv("DB_USER")
    _PASS = os.getenv("DB_PASS")
    _DB = os.getenv("DB_NAME")
    feature_names = (
        [f"mfcc{n}" for n in range(16)]
        + [f"mfcc_delta1{n}" for n in range(16)]
        + [f"mfcc_delta2{n}" for n in range(16)]
        + [
            "meanF0",
            "stdevF0",
            "meanF0delta",
            "hnr",
            "crest_factor",
            "rms",
            # "f_means",
            # "f_medians",
            "spectral_centroid",
            "spectral_rollof",
            "zero_crossing_rate",
        ]
    )

    def __init__(self, collection):
        self.COLLECTION = collection
        self.cluster = MongoClient(
            f"mongodb+srv://{self._USER}:{self._PASS}@cluster0.wywnr.mongodb.net/{self._DB}?retryWrites=true&w=majority"
        )
        self.self = self.cluster[self._DB]
        self.collection = self.self[collection]
        self._get_datasets()

    def post(self, file_name, label, features, augmentation=""):
        id = f"{file_name}_{augmentation}"

        if augmentation:
            aug = True
        else:
            aug = False
        post = {"_id": id, "label": label, "augmented": aug}
        post_features = dict(zip(self.feature_names, features))
        post.update(post_features)
        self.collection.insert_one(post)

    def select_by_id(self, id, like=False):
        if like:
            regx = re.compile(f".*{id}.*", re.IGNORECASE)
            out = self.collection.find_one({"_id": regx})
        else:
            out = self.collection.find_one({"_id": id})
        return out

    def select(self, field, rgx, multiple=True):
        rgx = re.compile(rgx, re.IGNORECASE)
        if multiple:
            out = self.collection.find({field: rgx})
        else:
            out = self.collection.find_one({field: rgx})
        return out

    def _get_datasets(self):
        dataset_no_aug = self.collection.find({"augmented": False})
        self.dataset_no_aug = pd.DataFrame(list(dataset_no_aug))
        dataset_full_aug = self.collection.find({})
        self.dataset_full_aug = pd.DataFrame(list(dataset_full_aug))
        dataset_balanced_aug = self.collection.find(
            {"$or": [{"label": 1}, {"augmented": False, "label": 0}]}
        )
        self.dataset_balanced_aug = pd.DataFrame(list(dataset_balanced_aug))

    def update(self):
        self._get_datasets()

    def print_balance(self):
        self.positives_no_aug = self.collection.count_documents(
            {"augmented": False, "label": 1}
        )
        self.negatives_no_aug = self.collection.count_documents(
            {"augmented": False, "label": 0}
        )

        self.positives_aug = self.collection.count_documents(
            {"augmented": True, "label": 1}
        )
        self.negatives_aug = self.collection.count_documents(
            {"augmented": True, "label": 0}
        )

        print(f"positives_no_aug: {self.positives_no_aug}")
        print(f"negatives_no_aug: {self.negatives_no_aug}")
        print(f"postives_aug: {self.positives_aug}")
        print(f"negatives_aug: {self.negatives_aug}")
        print(f"Total_positives: {self.positives_no_aug + self.positives_aug}")
        print(f"Total_negatives: {self.negatives_no_aug + self.negatives_aug}")

    def get_datasets(self):
        dataset_no_aug_y = self.dataset_no_aug["label"].to_numpy()
        dataset_no_aug_x = self.dataset_no_aug.drop(
            columns=["_id", "augmented", "label"]
        ).to_numpy()

        dataset_full_aug_y = self.dataset_full_aug["label"].to_numpy()
        dataset_full_aug_x = self.dataset_full_aug.drop(
            columns=["_id", "augmented", "label"]
        ).to_numpy()

        dataset_balanced_aug_y = self.dataset_balanced_aug["label"].to_numpy()
        dataset_balanced_aug_x = self.dataset_balanced_aug.drop(
            columns=["_id", "augmented", "label"]
        ).to_numpy()

        return (
            dataset_no_aug_x,
            dataset_no_aug_y,
            dataset_full_aug_x,
            dataset_full_aug_y,
            dataset_balanced_aug_x,
            dataset_balanced_aug_y,
        )


if __name__ == "__main__":

    db = Database("Dataset_bruto")

    db_val = Database("Dataset_validation")
    db_test = Database("Dataset_test")
    results_val = list(db_val.collection.find())
    results_test = list(db_test.collection.find())
    results = results_test + results_val
    db_train = Database("dataset_train")
    for n, doc in enumerate(results):
        results[n] = re.compile(f".*{doc['_id']}.*")

    results = list(db.collection.find({"_id": {"$nin": results}}))
    db_train.collection.insert_many(results)
