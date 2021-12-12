from itertools import cycle
from sklearn import metrics
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelBinarizer
from mlrun.artifacts import PlotArtifact
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp


def precision_recall_multi(ytest_b, yprob, labels):
    """"""
    n_classes = len(labels)

    precision = dict()
    recall = dict()
    avg_prec = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(
            ytest_b[:, i], yprob[:, i]
        )
        avg_prec[i] = metrics.average_precision_score(ytest_b[:, i], yprob[:, i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(
        ytest_b.ravel(), yprob.ravel()
    )
    avg_prec["micro"] = metrics.average_precision_score(ytest_b, yprob, average="micro")
    ap_micro = avg_prec["micro"]
    # model_metrics.update({'precision-micro-avg-classes': ap_micro})

    # gcf_clear(plt)
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
    plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall["micro"], precision["micro"], color="gold", lw=10)
    lines.append(l)
    labels.append(f"micro-average precision-recall (area = {ap_micro:0.2f})")

    for i, color in zip(range(n_classes), colors):
        (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(f"precision-recall for class {i} (area = {avg_prec[i]:0.2f})")

    # fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("precision recall - multiclass")
    plt.legend(lines, labels, loc=(0, -0.41), prop=dict(size=10))

    return PlotArtifact(
        "precision-recall-multiclass",
        body=plt.gcf(),
        title="Multiclass Precision Recall",
    )


def precision_recall_bin(model, xtest, ytest, yprob):
    """"""
    disp = metrics.plot_precision_recall_curve(model, xtest, ytest)
    disp.ax_.set_title(
        f"precision recall: AP={metrics.average_precision_score(ytest, yprob):0.2f}"
    )

    return PlotArtifact(
        "precision-recall-binary", body=disp.figure_, title="Binary Precision Recall"
    )


def roc_multi(ytest_b, yprob, labels):
    """"""
    n_classes = len(labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(ytest_b[:, i], yprob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ytest_b.ravel(), yprob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    gcf_clear(plt)
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (area = {roc_auc['micro']:0.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (area = {roc_auc['macro']:0.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("receiver operating characteristic - multiclass")
    plt.legend(loc=(0, -0.68), prop=dict(size=10))

    return PlotArtifact("roc-multiclass", body=plt.gcf(), title="Multiclass ROC Curve")



def roc_bin(ytest, yprob):
    """"""
    # ROC plot
    fpr, tpr, _ = metrics.roc_curve(ytest, yprob)
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="a label")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend(loc="best")

    return PlotArtifact("roc-binary", body=plt.gcf(), title="Binary ROC Curve")

def roc(context, model, ytest):
    is_multiclass = True if len(labels) > 2 else False

    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(xtest, **pred_params)

    if isinstance(ytest, np.ndarray):
        unique_labels = np.unique(ytest)
    elif isinstance(ytest, list):
        unique_labels = set(ytest)
    else:
        try:
            ytest = ytest.values
            unique_labels = np.unique(ytest)
        except Exception as exc:
            raise Exception(f"unrecognized data type for ytest {exc}")

    # AUC - ROC - PR CURVES
    if is_multiclass and is_classifier(model):
        lb = LabelBinarizer()
        ytest_b = lb.fit_transform(ytest)

        extra_data["precision_recall_multi"] = context.log_artifact(
            precision_recall_multi(ytest_b, yprob, unique_labels),
            artifact_path=plots_path,
            db_key=False,
        )
        extra_data["roc_multi"] = context.log_artifact(
            roc_multi(ytest_b, yprob, unique_labels),
            artifact_path=plots_path,
            db_key=False,
        )

        # AUC multiclass
        aucmicro = metrics.roc_auc_score(
            ytest_b, yprob, multi_class="ovo", average="micro"
        )
        aucweighted = metrics.roc_auc_score(
            ytest_b, yprob, multi_class="ovo", average="weighted"
        )

        context.log_results({"auc-micro": aucmicro, "auc-weighted": aucweighted})

        # others (todo - macro, micro...)
        f1 = metrics.f1_score(ytest, ypred, average="macro")
        ps = metrics.precision_score(ytest, ypred, average="macro")
        rs = metrics.recall_score(ytest, ypred, average="macro")
        context.log_results({"f1-score": f1, "precision_score": ps, "recall_score": rs})

    elif is_classifier(model):
        yprob_pos = yprob[:, 1]
        extra_data["precision_recall_bin"] = context.log_artifact(
            precision_recall_bin(model, xtest, ytest, yprob_pos),
            artifact_path=plots_path,
            db_key=False,
        )
        extra_data["roc_bin"] = context.log_artifact(
            roc_bin(ytest, yprob_pos, clear=True),
            artifact_path=plots_path,
            db_key=False,
        )

        rocauc = metrics.roc_auc_score(ytest, yprob_pos)
        brier_score = metrics.brier_score_loss(ytest, yprob_pos, pos_label=ytest.max())
        f1 = metrics.f1_score(ytest, ypred)
        ps = metrics.precision_score(ytest, ypred)
        rs = metrics.recall_score(ytest, ypred)
        context.log_results(
            {
                "rocauc": rocauc,
                "brier_score": brier_score,
                "f1-score": f1,
                "precision_score": ps,
                "recall_score": rs,
            }
        )

    elif is_regressor(model):
        r_squared = r2_score(ytest, ypred)
        rmse = mean_squared_error(ytest, ypred, squared=False)
        mse = mean_squared_error(ytest, ypred, squared=True)
        mae = mean_absolute_error(ytest, ypred)
        context.log_results(
            {
                "R2": r_squared,
                "root_mean_squared_error": rmse,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
            }
        )
    # return all model metrics and plots
    return extra_data