
import mlflow
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from mlflow.models import infer_signature

from data import (
    COMMON_TAGS,
    train_ds,
    X_train,
    y_train
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["recall", "precision", "f1", "roc_auc"]


def run_experiment(run_name, pipe, tags=None, note=""):
    with mlflow.start_run(run_name=run_name) as run:

        # (a) KONTEKS: tag, catatan hipotesis, versi data
        mlflow.set_tags({
            **COMMON_TAGS,
            "candidate": "true",
            **(tags or {})
        })
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_input(train_ds, context="training")

        # (b) PARAMETER: pipeline dan hyperparameter estimator
        mlflow.log_param(
            "pipeline_steps",
            " -> ".join(pipe.named_steps)
        )
        mlflow.log_params({
            f"clf__{k}": v
            for k, v in pipe[-1].get_params().items()
        })

        # (c) METRIK: mean dan standar deviasi dari 5-fold CV
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING
        )

        for m in SCORING:
            mlflow.log_metric(
                f"cv_{m}_mean",
                cv[f"test_{m}"].mean()
            )
            mlflow.log_metric(
                f"cv_{m}_std",
                cv[f"test_{m}"].std()
            )

        mlflow.log_metric("fit_time_s", cv["fit_time"].mean())

        # (d) ARTEFAK DIAGNOSTIK: prediksi out-of-fold
        y_oof = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=CV
        )

        disp = ConfusionMatrixDisplay.from_predictions(
            y_train,
            y_oof,
            display_labels=["benign", "malignant"]
        )

        mlflow.log_figure(
            disp.figure_,
            "plots/confusion_matrix_oof.png"
        )
        plt.close(disp.figure_)

        mlflow.log_text(
            classification_report(
                y_train,
                y_oof,
                zero_division=0
            ),
            "reports/classification_report_oof.txt"
        )

        # (e) MODEL: latih ulang pada seluruh data train
        pipe.fit(X_train, y_train)

        signature = infer_signature(
            X_train,
            pipe.predict(X_train)
        )

        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        mlflow.set_tag("model_uri", info.model_uri)

        return run.info.run_id
