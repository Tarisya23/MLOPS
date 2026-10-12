
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from tracking import run_experiment


# Eksperimen 00: baseline tanpa model pembelajaran
run_experiment(
    "00-dummy-most-frequent",
    Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent"))
    ]),
    tags={"stage": "baseline", "model_family": "dummy"},
    note="Batas bawah: selalu menebak kelas mayoritas (benign)."
)


# Eksperimen 01: Logistic Regression tanpa scaling
run_experiment(
    "01-logreg-noscale",
    Pipeline([
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    tags={"stage": "baseline", "model_family": "logreg"},
    note="H1: model linear tanpa preprocessing."
)


# Eksperimen 02: Logistic Regression dengan scaling
run_experiment(
    "02-logreg-scaled",
    Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    tags={"stage": "baseline", "model_family": "logreg"},
    note="H2: hanya menambah StandardScaler, variabel lain tetap."
)

print("Tiga eksperimen baseline selesai dijalankan.")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

candidates = {
    "knn": KNeighborsClassifier(n_neighbors=5),
    "tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    "gb": GradientBoostingClassifier(random_state=42),
    "svc": SVC(kernel="rbf", random_state=42),
}

for i, (family, clf) in enumerate(candidates.items(), start=3):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    run_experiment(
        f"{i:02d}-{family}-default",
        pipe,
        tags={"stage": "model-selection", "model_family": family},
        note="Hyperparameter default; preprocessing identik untuk semua."
    )
