import  os, warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

feats = ["mean", "std_dev", "skewness", "kurtosis",
         "glcm_contrast", "glcm_correlation", "glcm_energy",
         "glcm_entropy", "glcm_homogeneity", "glcm_cluster_shade"]

TASKS = {
    "susceptibility": {
        "col": "maoe_susceptibility",
        "classes": ["Normal (0)", "Abnormal (1)"],
        "scoring": {"rec": "recall", "prec": "precision", "f1": "f1", "auc": "roc_auc"},
        "auc_mode": "binary",
        "paper": {"acc": 92.94, "rec": 94.00, "prec": 92.04, "f1": 93.01, "auc": 97.99},
    },
    "risk_acuity": {
        "col": "maoe_risk",
        "classes": ["No risk (0)", "Low risk (1)", "High risk (2)"],
        "scoring": {"rec": "recall_macro", "prec": "precision_macro",
                     "f1": "f1_macro", "auc": "roc_auc_ovr_weighted"},
        "auc_mode": "ovr_weighted",
        "paper": {"acc": 95.02, "rec": 95.84, "prec": 95.84, "f1": 95.84, "auc": 99.45},
    },
}

SEED = 42


def etc():
    return ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
        min_samples_leaf=1, min_samples_split=2, criterion="gini",
        random_state=SEED, n_jobs=-1)

def rf():
    return RandomForestClassifier(n_estimators=500, max_features="sqrt",
        random_state=SEED, n_jobs=-1)

def lr(w):
    return LogisticRegression(C=1.0, class_weight=w, max_iter=1000,
        random_state=SEED)


def get_scores(yt, yp, yprob, classes, auc_mode):
    binary = len(classes) == 2
    avg = "binary" if binary else "macro"
    acc = accuracy_score(yt, yp) * 100
    rec = recall_score(yt, yp, average=avg, zero_division=0) * 100
    pr = precision_score(yt, yp, average=avg, zero_division=0) * 100
    f1 = f1_score(yt, yp, average=avg, zero_division=0) * 100
    kappa = cohen_kappa_score(yt, yp) * 100
    mcc = matthews_corrcoef(yt, yp) * 100
    try:
        if auc_mode == "binary":
            auc = roc_auc_score(yt, yprob[:, 1]) * 100
        else:
            auc = roc_auc_score(yt, yprob, multi_class="ovr", average="weighted") * 100
    except:
        auc = 0.0
    return {"acc": acc, "rec": rec, "prec": pr, "f1": f1,
            "auc": auc, "kappa": kappa, "mcc": mcc}


def paper_exact_method(xtr, ytr, xte, yte, classes, auc_mode):
    x = np.vstack([xtr, xte])
    y = np.concatenate([ytr, yte])
    xr, yr = SMOTE(random_state=SEED).fit_resample(x, y)
    xtrain, xtest, ytrain, ytest = train_test_split(
        xr, yr, test_size=0.25, stratify=yr, random_state=SEED)
    m = etc()
    m.fit(xtrain, ytrain)
    pred = m.predict(xtest)
    prob = m.predict_proba(xtest)
    return get_scores(ytest, pred, prob, classes, auc_mode)


def fixed_etc_adasyn(xtr, ytr, xte, yte, classes, auc_mode):
    sc = MinMaxScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)
    xr, yr = ADASYN(random_state=SEED).fit_resample(xtr_s, ytr)
    m = etc()
    m.fit(xr, yr)
    pred = m.predict(xte_s)
    prob = m.predict_proba(xte_s)
    return get_scores(yte, pred, prob, classes, auc_mode)


def rf_with_adasyn(xtr, ytr, xte, yte, classes, auc_mode):
    sc = MinMaxScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)
    xr, yr = ADASYN(random_state=SEED).fit_resample(xtr_s, ytr)
    m = rf()
    m.fit(xr, yr)
    pred = m.predict(xte_s)
    prob = m.predict_proba(xte_s)
    return get_scores(yte, pred, prob, classes, auc_mode)


def lr_balanced(xtr, ytr, xte, yte, classes, auc_mode):
    sc = MinMaxScaler()
    xtr_s = sc.fit_transform(xtr)
    xte_s = sc.transform(xte)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    w = dict(enumerate(cw))
    m = lr(w)
    m.fit(xtr_s, ytr)
    pred = m.predict(xte_s)
    prob = m.predict_proba(xte_s)
    return get_scores(yte, pred, prob, classes, auc_mode)


def run_comparison(task, cfg, d):
    col = cfg["col"]
    classes = cfg["classes"]
    auc_mode = cfg["auc_mode"]
    paper = cfg["paper"]

    df1 = pd.read_csv(os.path.join(d, "features_train.csv"))
    df2 = pd.read_csv(os.path.join(d, "features_test.csv"))

    cols = list(feats)
    if "glcm_cluster_shade" not in df1.columns and "glcm_idm" in df1.columns:
        cols = [c if c != "glcm_cluster_shade" else "glcm_idm" for c in cols]

    xtr = df1[cols].values.astype(np.float64)
    ytr = df1[col].values.astype(int)
    xte = df2[cols].values.astype(np.float64)
    yte = df2[col].values.astype(int)

    methods = {
        "Paper method (SMOTE leaky)": paper_exact_method,
        "ETC + ADASYN (fixed)": fixed_etc_adasyn,
        "RandomForest + ADASYN": rf_with_adasyn,
        "LogReg + class_weight": lr_balanced,
    }

    results = {}
    for name, fn in methods.items():
        results[name] = fn(xtr, ytr, xte, yte, classes, auc_mode)

    print(f"\n{'='*95}")
    print(f"  {task.upper()} - Method Comparison")
    print(f"{'='*95}")
    print(f"  {'Method':<30} {'Acc':>7} {'Rec':>7} {'Prec':>7} {'F1':>7} {'AUC':>7} {'Kappa':>7} {'MCC':>7}")
    print(f"  {'-'*93}")

    print(f"  {'[Paper reported]':<30} {paper['acc']:>6.2f}% {paper['rec']:>6.2f}%"
          f" {paper['prec']:>6.2f}% {paper['f1']:>6.2f}% {paper['auc']:>6.2f}%"
          f" {'--':>7} {'--':>7}")

    for name, s in results.items():
        print(f"  {name:<30} {s['acc']:>6.2f}% {s['rec']:>6.2f}%"
              f" {s['prec']:>6.2f}% {s['f1']:>6.2f}% {s['auc']:>6.2f}%"
              f" {s['kappa']:>6.2f}% {s['mcc']:>6.2f}%")

    print(f"  {'-'*93}")
    print()


def main():
    

    for name, cfg in TASKS.items():
        run_comparison(name, cfg, '.')

    print("done")


if __name__ == "__main__":
    main()
