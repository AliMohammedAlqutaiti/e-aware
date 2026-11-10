from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path("data/imarpe/E_AWARE_IMARPE_training_ready.csv")
OUTPUT = Path("data/imarpe/E_AWARE_IMARPE_occurrence_predictions.csv")

FEATURES = [
    "sst_c",
    "sss_psu",
    "oxygen_mg_l",
    "bottom_depth_m",
]

TARGET = "presence"
GROUP = "cruise"


def main():
    df = pd.read_csv(INPUT)

    all_predictions = []

    print("==========================================")
    print("E-AWARE IMARPE OCCURRENCE BASELINE")
    print("Leave-One-Cruise-Out Logistic Regression")
    print("==========================================")

    for cruise in sorted(df[GROUP].unique()):
        train = df[df[GROUP] != cruise].copy()
        test = df[df[GROUP] == cruise].copy()

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "numeric",
                                Pipeline(
                                    steps=[
                                        (
                                            "imputer",
                                            SimpleImputer(strategy="median"),
                                        ),
                                        (
                                            "scaler",
                                            StandardScaler(),
                                        ),
                                    ]
                                ),
                                FEATURES,
                            )
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold = test[
            [
                "record_id",
                "cruise",
                "date",
                "lat",
                "lon",
                "anchoveta_acoustic",
                "presence",
            ]
        ].copy()

        fold["predicted_probability"] = prob
        fold["predicted_presence"] = pred

        all_predictions.append(fold)

        print(
            f"Cruise {cruise}: "
            f"test_rows={len(test)}, "
            f"observed_presence={int(y_test.sum())}, "
            f"mean_predicted_probability={prob.mean():.3f}"
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    y_true = predictions["presence"].astype(int)
    y_prob = predictions["predicted_probability"]
    y_pred = predictions["predicted_presence"].astype(int)

    print("\n==========================================")
    print("OUT-OF-CRUISE VALIDATION RESULTS")
    print("==========================================")

    print("Rows:", len(predictions))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_prob), 3))
    print(
        "PR-AUC:",
        round(
            average_precision_score(y_true, y_prob),
            3,
        ),
    )
    print(
        "Brier score:",
        round(
            brier_score_loss(y_true, y_prob),
            3,
        ),
    )
    print(
        "Accuracy:",
        round(
            accuracy_score(y_true, y_pred),
            3,
        ),
    )
    print(
        "Precision:",
        round(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            3,
        ),
    )
    print(
        "Recall:",
        round(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            3,
        ),
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    predictions.to_csv(OUTPUT, index=False)

    print("\nSaved predictions:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
