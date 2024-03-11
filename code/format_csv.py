import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

gesture_list = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up", "No gesture"]

train_df = pd.read_csv("./annotations/train.csv", sep=";", names=("id", "label"))
train_df = train_df[train_df["label"].isin(gesture_list)]
train_df.to_csv("./annotations/train_formatted.csv", sep=";", index=False)

val_df = pd.read_csv("./annotations/validation.csv", sep=";", names=("id", "label"))
val_df = val_df[val_df["label"].isin(gesture_list)]
val_df.to_csv("./annotations/validation_formatted.csv", sep=";", index=False)