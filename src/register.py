
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()
NAME = "bc-malignant-classifier"
EXPERIMENT_NAME = "bc-malignant-detection"

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError("Eksperimen tidak ditemukan.")

# Ambil parent run hasil GridSearchCV
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    max_results=1000
)

candidates = [
    run for run in runs
    if run.data.tags.get("mlflow.runName") == "10-svc-gridsearch"
]

if not candidates:
    raise ValueError("Run 10-svc-gridsearch tidak ditemukan.")

best_run = candidates[0]
best_model_uri = best_run.data.tags.get("model_uri")

if not best_model_uri:
    raise ValueError(
        "Model URI kosong. Pastikan model hasil tuning sudah tercatat di MLflow."
    )

# Pastikan evaluasi test set sudah dilakukan
if best_run.data.tags.get("test_evaluated") != "true":
    raise ValueError(
        "Evaluasi test set belum tercatat. Jalankan tahap L6 terlebih dahulu."
    )

test_recall = best_run.data.metrics.get("test_recall")
test_precision = best_run.data.metrics.get("test_precision")

if test_recall is None or test_precision is None:
    raise ValueError("Metrik evaluasi test set tidak ditemukan.")

data_md5 = best_run.data.tags.get("data_md5", "unknown")

# Daftarkan model ke Model Registry
mv = mlflow.register_model(best_model_uri, NAME)

client.update_model_version(
    NAME,
    mv.version,
    description=(
        "Model SVC hasil GridSearchCV; "
        "dipilih berdasarkan recall CV dengan precision minimal 0.90."
    )
)

client.set_model_version_tag(
    NAME, mv.version, "data_md5", data_md5
)

client.set_registered_model_alias(
    NAME, "challenger", mv.version
)

# Quality gate untuk promosi menjadi champion
MIN_PRECISION = 0.90
MARGIN = 0.005

def champion_recall():
    try:
        champ = client.get_model_version_by_alias(NAME, "champion")
        champ_run = client.get_run(champ.run_id)
        return champ_run.data.metrics.get("test_recall", -1.0)
    except Exception:
        return -1.0

passed = (
    test_precision >= MIN_PRECISION
    and test_recall >= champion_recall() + MARGIN
)

status = "approved" if passed else "rejected"

client.set_model_version_tag(
    NAME, mv.version, "validation_status", status
)

if passed:
    client.set_registered_model_alias(
        NAME, "champion", mv.version
    )

print("Nama model       :", NAME)
print("Versi            :", mv.version)
print("Test recall      :", round(test_recall, 4))
print("Test precision   :", round(test_precision, 4))
print("Validation status:", status)

if passed:
    print("Model berhasil dipromosikan menjadi champion.")
else:
    print("Model belum memenuhi syarat promosi menjadi champion.")
