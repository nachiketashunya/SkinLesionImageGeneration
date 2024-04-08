import pandas as pd


TRAIN_LABELS = "data/Assignment_4/Train/Train_labels.csv"

labels = pd.read_csv(TRAIN_LABELS)
print(labels.head())