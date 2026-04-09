import sys

sys.path.append("..")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)


def visualize(model, feature_set, X_test, Y_test, Y_pred):
    if hasattr(model, "predict_proba"):
        Y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        Y_score = model.decision_function(X_test)
    else:
        Y_score = None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay.from_predictions(
        Y_test, Y_pred, ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title(f"Confusion Matrix: {feature_set}")

    RocCurveDisplay.from_predictions(Y_test, Y_score, ax=axes[1])
    roc_auc = roc_auc_score(Y_test, Y_score)
    axes[1].set_title(f"ROC Curve: {feature_set} (AUC = {roc_auc:.3f})")

    PrecisionRecallDisplay.from_predictions(Y_test, Y_score, ax=axes[2])
    ap = average_precision_score(Y_test, Y_score)
    axes[2].set_title(f"Precision-Recall Curve: {feature_set} (AP = {ap:.3f})")

    plt.tight_layout()
    plt.show()

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
