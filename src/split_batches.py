import pandas as pd
import numpy as np
import os

os.makedirs("data/raw/train", exist_ok=True)
os.makedirs("data/raw/test", exist_ok=True)

df = pd.read_csv("data/raw/train.csv")

df = df.sort_values(
    by=["Departure Delay in Minutes", "Arrival Delay in Minutes"]
)

sizes = [30000, 30000, 30000]

batch_1 = df.iloc[:sizes[0]]
batch_2 = df.iloc[sizes[0]:sum(sizes[:2])]
batch_3 = df.iloc[sum(sizes[:2]):sum(sizes)]
batch_4 = df.iloc[sum(sizes):]

batch_3["satisfaction"] = np.where(
    np.random.rand(len(batch_3)) < 0.10,
    "neutral or dissatisfied",
    batch_3["satisfaction"]
)

batch_4["satisfaction"] = np.where(
    np.random.rand(len(batch_4)) < 0.20,
    "neutral or dissatisfied",
    batch_4["satisfaction"]
)

batch_1.to_csv("data/raw/train/batch_1.csv", index=False)
batch_2.to_csv("data/raw/train/batch_2.csv", index=False)
batch_3.to_csv("data/raw/train/batch_3.csv", index=False)
batch_4.to_csv("data/raw/train/batch_4.csv", index=False)

pd.read_csv("data/raw/test.csv").to_csv(
    "data/raw/test/batch_test.csv", index=False
)

print("Batches created successfully")
