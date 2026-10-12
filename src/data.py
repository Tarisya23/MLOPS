
import hashlib
import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

mlflow.set_experiment("bc-malignant-detection")

df = load_breast_cancer(as_frame=True).frame
df["target"] = 1 - df["target"]  # 1 = malignant (kelas positif)

# Sidik jari data: berubah satu nilai pun → hash berbeda
data_md5 = hashlib.md5(
    pd.util.hash_pandas_object(df, index=True).values
).hexdigest()

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["target"],
    random_state=42
)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

X_train = train_df.drop(columns="target")
y_train = train_df["target"]

X_test = test_df.drop(columns="target")
y_test = test_df["target"]

train_ds = mlflow.data.from_pandas(
    train_df,
    source="data/train.csv",
    name="bc-train",
    targets="target"
)

COMMON_TAGS = {
    "data_md5": data_md5,
    "split_seed": "42",
    "dataset": "sklearn-breast-cancer"
}
