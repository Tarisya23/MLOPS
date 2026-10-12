
import mlflow
import pandas as pd

mlflow.set_experiment("bc-malignant-detection")

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(
    "bc-malignant-detection"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    max_results=1000
)

records = []

for run in runs:
    metrics = run.data.metrics
    tags = run.data.tags

    recall = metrics.get("cv_recall_mean")
    precision = metrics.get("cv_precision_mean")

    if recall is None or precision is None:
        continue

    records.append({
        "run_id": run.info.run_id,
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "stage": tags.get("stage", ""),
        "model_family": tags.get("model_family", ""),
        "recall": recall,
        "precision": precision,
        "f1": metrics.get("cv_f1_mean"),
        "roc_auc": metrics.get("cv_roc_auc_mean"),
        "model_uri": tags.get("model_uri", "")
    })

results = pd.DataFrame(records)

if results.empty:
    raise ValueError(
        "Tidak ditemukan run dengan metrik CV recall dan precision."
    )

# Quality gate: precision minimal 0.90
eligible = results[results["precision"] >= 0.90].copy()

if eligible.empty:
    raise ValueError(
        "Tidak ada model yang memenuhi syarat precision >= 0.90."
    )

eligible = eligible.sort_values(
    by="recall",
    ascending=False
)

print("\n=== Semua kandidat ===")
print(
    results[
        ["run_name", "model_family", "recall", "precision", "f1", "roc_auc"]
    ].sort_values("recall", ascending=False).to_string(index=False)
)

print("\n=== Kandidat yang lolos quality gate ===")
print(
    eligible[
        ["run_name", "model_family", "recall", "precision", "f1", "roc_auc"]
    ].to_string(index=False)
)

best = eligible.iloc[0]

print("\n=== Model terpilih ===")
print("Nama run :", best["run_name"])
print("Keluarga model :", best["model_family"])
print("Recall CV :", best["recall"])
print("Precision CV :", best["precision"])
print("Model URI :", best["model_uri"])

eligible.to_csv("data/eligible_models.csv", index=False)
print("\nDaftar kandidat tersimpan di data/eligible_models.csv")


import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# Menghubungkan ke MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(
    "bc-malignant-detection"
)

# Mengambil parent run hasil GridSearchCV
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    max_results=1000
)

parent = [
    run for run in runs
    if run.data.tags.get("mlflow.runName") == "10-svc-gridsearch"
]

if not parent:
    raise ValueError("Run 10-svc-gridsearch tidak ditemukan.")

best_run = parent[0]
best_run_id = best_run.info.run_id
best_model_uri = best_run.data.tags.get("model_uri")

if not best_model_uri:
    raise ValueError(
        "Model URI pada 10-svc-gridsearch kosong. "
        "Periksa apakah model telah dicatat menggunakan mlflow.sklearn.log_model."
    )

# Memuat data uji yang sebelumnya disisihkan
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns="target")
y_test = test_df["target"]

# Memuat model hasil tuning
model = mlflow.sklearn.load_model(best_model_uri)

y_pred = model.predict(X_test)

score = (
    model.predict_proba(X_test)[:, 1]
    if hasattr(model, "predict_proba")
    else model.decision_function(X_test)
)

test_metrics = {
    "test_recall": recall_score(y_test, y_pred),
    "test_precision": precision_score(y_test, y_pred),
    "test_f1": f1_score(y_test, y_pred),
    "test_roc_auc": roc_auc_score(y_test, score),
}

print("=== Hasil Evaluasi Test Set ===")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Mencatat hasil evaluasi pada run yang sama
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metrics(test_metrics)

    mlflow.log_input(
        mlflow.data.from_pandas(
            test_df,
            source="data/test.csv",
            name="bc-test",
            targets="target"
        ),
        context="testing"
    )

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["benign", "malignant"]
    )

    mlflow.log_figure(
        disp.figure_,
        "plots/confusion_matrix_test.png"
    )
    plt.close(disp.figure_)

    mlflow.set_tag("test_evaluated", "true")

print("\nEvaluasi akhir berhasil dicatat di MLflow.")
