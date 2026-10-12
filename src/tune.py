
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from mlflow.models import infer_signature

from data import COMMON_TAGS, train_ds, X_train, y_train
from tracking import CV, SCORING


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", random_state=42))
])

grid = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.01, 0.001]
}

gs = GridSearchCV(
    pipe,
    grid,
    cv=CV,
    scoring=SCORING,
    refit="recall",
    n_jobs=-1
)

with mlflow.start_run(
    run_name="10-svc-gridsearch",
    tags={
        **COMMON_TAGS,
        "stage": "tuning",
        "model_family": "svc",
        "candidate": "true"
    }
):
    mlflow.log_input(train_ds, context="training")
    mlflow.log_dict(grid, "search_space.json")

    gs.fit(X_train, y_train)
    res = gs.cv_results_

    # Setiap kombinasi parameter dicatat sebagai child run
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(
            run_name=f"svc-trial-{i:02d}",
            nested=True
        ):
            mlflow.log_params(params)

            mlflow.log_metrics({
                f"cv_{m}_mean": res[f"mean_test_{m}"][i]
                for m in SCORING
            })

            mlflow.log_metrics({
                f"cv_{m}_std": res[f"std_test_{m}"][i]
                for m in SCORING
            })

    # Catat parameter dan metrik kandidat terbaik pada parent run
    b = gs.best_index_

    mlflow.log_params(gs.best_params_)

    mlflow.log_metrics({
        f"cv_{m}_mean": res[f"mean_test_{m}"][b]
        for m in SCORING
    })

    mlflow.log_metrics({
        f"cv_{m}_std": res[f"std_test_{m}"][b]
        for m in SCORING
    })

    info = mlflow.sklearn.log_model(
        gs.best_estimator_,
        name="model",
        signature=infer_signature(
            X_train,
            gs.best_estimator_.predict(X_train)
        ),
        input_example=X_train.head(3)
    )

    mlflow.set_tag("model_uri", info.model_uri)

    print("Parameter terbaik:", gs.best_params_)
    print("Recall CV terbaik:", res["mean_test_recall"][b])
    print("Precision CV:", res["mean_test_precision"][b])
    print("F1-score CV:", res["mean_test_f1"][b])
    print("ROC-AUC CV:", res["mean_test_roc_auc"][b])
