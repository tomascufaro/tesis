import os
from dotenv import load_dotenv
import re

import pymongo
from pymongo import MongoClient

import pandas as pd
import numpy as np


class Database:
    load_dotenv()

    __USER = os.getenv("DB_USER")
    __PASS = os.getenv("DB_PASS")
    __DB = os.getenv("DB_NAME")
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
            f"mongodb+srv://{self.__USER}:{self.__PASS}@cluster0.wywnr.mongodb.net/{self.__DB}?retryWrites=true&w=majority"
        )
        self.self = self.cluster[self.__DB]
        self.collection = self.self[collection]
        self.__get_datasets()

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

    def __get_datasets(self):
        try:
            dataset_no_aug = self.collection.find({"augmented": False})
            self.dataset_no_aug = pd.DataFrame(list(dataset_no_aug))
            dataset_full_aug = self.collection.find({})
            self.dataset_full_aug = pd.DataFrame(list(dataset_full_aug))
            dataset_balanced_aug = self.collection.find(
                {"$or": [{"label": 1}, {"augmented": False, "label": 0}]}
            )
            self.dataset_balanced_aug = pd.DataFrame(list(dataset_balanced_aug))
        except:
            print("Empty collection")

    def update(self):
        self.__get_datasets()

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
    Database("IEMOCAP")
