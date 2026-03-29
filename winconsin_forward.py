import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

# =========================
# Load Dataset
# =========================

data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# =========================
# Scale
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Metric Dictionary (เร็วกว่า if-elif)
# =========================

metric_funcs = {

    'accuracy':
        lambda y_t, y_p:
            accuracy_score(y_t, y_p),

    'recall':
        lambda y_t, y_p:
            recall_score(
                y_t,
                y_p,
                average='macro',
                zero_division=0
            ),

    'f1_score':
        lambda y_t, y_p:
            f1_score(
                y_t,
                y_p,
                average='macro',
                zero_division=0
            ),

    'precision':
        lambda y_t, y_p:
            precision_score(
                y_t,
                y_p,
                average='macro',
                zero_division=0
            )
}

# =========================
# Forward Selection (Optimized)
# =========================

def forward_selection(
    model,
    metric_name,
    max_features=15
):

    selected = []
    best_global_score = -np.inf

    while len(selected) < max_features:

        best_score = -np.inf
        best_feature = None

        remaining = list(
            set(range(X_train.shape[1]))
            - set(selected)
        )

        for f in remaining:

            trial = selected + [f]

            clf = clone(model)

            clf.fit(
                X_train[:, trial],
                y_train
            )

            y_pred = clf.predict(
                X_test[:, trial]
            )

            score = metric_funcs[
                metric_name
            ](y_test, y_pred)

            if score > best_score:

                best_score = score
                best_feature = f

        if best_score > best_global_score:

            selected.append(best_feature)
            best_global_score = best_score

            print(
                f"[Forward-{metric_name}] "
                f"Added: {feature_names[best_feature]} "
                f"| Score: {best_global_score:.4f}"
            )

        else:
            print("[Forward] Stop")
            break

    return selected


# =========================
# Backward Selection (Optimized)
# =========================

def backward_selection(
    model,
    metric_name,
    initial_k=15,
    threshold_ratio=0.95
):

    selected = list(range(initial_k))

    clf = clone(model)

    clf.fit(
        X_train[:, selected],
        y_train
    )

    y_pred = clf.predict(
        X_test[:, selected]
    )

    best_score = metric_funcs[
        metric_name
    ](y_test, y_pred)

    threshold = threshold_ratio * best_score

    print(
        f"[Backward-{metric_name}] "
        f"Initial Score: {best_score:.4f}"
    )

    while len(selected) > 1:

        best_score_round = -np.inf
        remove_feature = None

        for f in selected:

            trial = [
                x for x in selected
                if x != f
            ]

            clf = clone(model)

            clf.fit(
                X_train[:, trial],
                y_train
            )

            y_pred = clf.predict(
                X_test[:, trial]
            )

            score = metric_funcs[
                metric_name
            ](y_test, y_pred)

            if score > best_score_round:

                best_score_round = score
                remove_feature = f

        if best_score_round >= threshold:

            selected.remove(remove_feature)

            print(
                f"[Backward-{metric_name}] "
                f"Removed: {feature_names[remove_feature]} "
                f"| Score: {best_score_round:.4f}"
            )

        else:
            print("[Backward] Stop")
            break

    return selected


# =========================
# Models
# =========================

models = {

    'Decision Tree':
        DecisionTreeClassifier(
            random_state=42
        ),

    'Random Forest':
        RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,  # ใช้ CPU ทุก core
            random_state=42
        ),

    'KNN':
        KNeighborsClassifier(
            n_neighbors=5
        ),

    'Naive Bayes':
        GaussianNB(),

    'Perceptron':
        Perceptron(
            random_state=42
        ),

    'Logistic Regression':
        LogisticRegression(
            solver='liblinear'
        )
}

# =========================
# Metrics
# =========================

metrics = [

    'accuracy',
    'recall',
    'f1_score',
    'precision'
]

# =========================
# Run
# =========================

for metric in metrics:

    print("\n====================")
    print("METRIC:", metric)
    print("====================")

    for name, model in models.items():

        print(f"\nMODEL: {name}")

        print("\n--- Forward ---")

        f_sel = forward_selection(
            model,
            metric
        )

        print(
            f"\nForward Selected ({len(f_sel)})"
        )

        print(feature_names[f_sel])

        print("\n--- Backward ---")

        b_sel = backward_selection(
            model,
            metric
        )

        print(
            f"\nBackward Selected ({len(b_sel)})"
        )

        print(feature_names[b_sel])