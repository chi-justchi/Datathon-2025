"""
modified from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py
"""
from pathlib import Path

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MimicDataset(Dataset):

    split_ratio = [0.6, 0.2, 0.2]

    embed_prefix = "embds"

    pathologies = ["Enlarged Cardiomediastinum",
                   "Cardiomegaly",
                   "Lung Opacity",
                   "Lung Lesion",
                   "Edema",
                   "Consolidation",
                   "Pneumonia",
                   "Atelectasis",
                   "Pneumothorax",
                   "Pleural Effusion",
                   "Pleural Other",
                   "Fracture",
                   "Support Devices"]


    embedding_d = {
        "BiomedCLIP": Path("~/fsx/embeddings/MIMIC/embds_BiomedCLIP"), 
        "CheXagent": Path("~/fsx/embeddings/MIMIC/embds_CheXagent"), 
        "MedGemma": Path("~/fsx/embeddings/MIMIC/embds_MedGemma"),
        "MedImageInsights": Path("~/fsx/embeddings/MIMIC/embds_MedGemma"),
        "RAD-DINO": Path("~/fsx/embeddings/MIMIC/embds_MedGemma"),
    }

    csvpath = Path("~/fsx/embeddings/MIMIC/Tables/mimic-cxr-2.0.0-chexpert.csv")
    metacsvpath = Path("~/fsx/embeddings/MIMIC/Tables/mimic-cxr-2.0.0-metadata.csv")
    base_dicom_path = Path("~/fsx/embeddings/MIMIC/")

    def __init__(
        self,
        views: str = ["PA", "AP"][0],
        mode: str = ["train", "validate", "test"][0],
        embedding_type: str = ["BiomedCLIP", "CheXagent", "MedGemma", "MedImageInsights", "RAD-DINO", "All"][0],       
        unique_patients=True,
        transform = None,
        seed : int = 0):
        
        np.random.seed(seed)  # Reset the seed so all runs are the same.        
        self.views = views
        self.mode = mode
        self.embedding_type = embedding_type
        self.unique_patients = unique_patients
        self.transform = transform
        self.seed = seed
        
        self.embpath: str | list[str] = self.load_emb_path(embedding_type)
            
        self.csv = pd.read_csv(self.csvpath)
        self.metacsv = pd.read_csv(self.metacsvpath)
        self.csv = self.csv.set_index(["subject_id", "study_id"])
        self.metacsv = self.metacsv.set_index(["subject_id", "study_id"])
        self.csv = self.csv.join(self.metacsv).reset_index()        

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)
    
        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        self.csv = self.csv.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.csv = self.csv.fillna(0)
                    
        n_row = self.csv.shape[0]

        # spit data to one of train valid test
        if self.mode == "train":
            self.csv = self.csv[: int(n_row * self.split_ratio[0])]
        elif self.mode == "valid":
            self.csv = self.csv[
                int(n_row * self.split_ratio[0]) : int(
                    n_row * (self.split_ratio[0] + self.split_ratio[1])
                )
            ]
        elif self.mode == "test":
            self.csv = self.csv[-int(n_row * self.split_ratio[-1]) :]
        else:
            raise ValueError(
                f"attr:mode has to be one of [train, valid, test] but your input is {self.mode}"
            )

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = 0

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patient_id"] = self.csv["subject_id"].astype(str)

    def __getitem__(self, i):

        sample = {}
        sample["patient_id"] = int(self.csv["patient_id"][i])
        sample["case_id"] = int(self.csv["patient_id"][i])
        sample["lab"] = self.labels[i]
        sample["emb"] = self.load_embedding(self.csv["dicom_id"][i])

        return sample

    def __len__(self):
        return len(self.labels)        
        

    def load_emb_path(self, embedding_type):
        if self.embedding_type != "All":
            return self.embedding_d[embedding_type]
        else:            
            return list(self.embedding_d.values())

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view
    
    def load_embedding(self, embedding_id):
        return np.load(f"{self.base_dicom_path/"embds_"+ self.embedding_type/embedding_id}.npy")
    
if __name__ == "__main__":


     dataset = MimicDataset(
        views="PA",
        mode="train",
        embedding_type="BiomedCLIP",
        unique_patients=True,
        transform=None,        
        seed=0
    )# Datathon-2025
