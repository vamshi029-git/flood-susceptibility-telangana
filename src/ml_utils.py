"""Feature preparation, the 7 models and evaluation metrics."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import CAT, NUM, SEED


def add_features(df):
    """Aspect is circular (0° = 360°), so represent it as sin / cos."""
    df = df.copy()
    df['aspect_sin'] = np.sin(np.deg2rad(df['aspect']))
    df['aspect_cos'] = np.cos(np.deg2rad(df['aspect']))
    return df


def usable_numeric(df, num=NUM):
    """Drop numeric factors that are missing or constant (they carry no information)."""
    keep = [c for c in num if c in df.columns and df[c].std() > 0]
    dropped = [c for c in num if c not in keep]
    if dropped:
        print('Dropped (missing or constant):', dropped)
    return keep


def make_prep(num, cat=CAT):
    return ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
    ])


def get_models():
    # LR / SVM / RF / XGBoost use the base paper's tuned hyperparameters (Tepetidis et al., 2025)
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    rf = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt',
                                min_samples_split=2, min_samples_leaf=1,
                                random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, subsample=1.0,
                        colsample_bytree=0.8, eval_metric='logloss', random_state=SEED)
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
    return {
        'LR': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='rbf', C=2, gamma='scale', probability=True, random_state=SEED),
        'RF': rf,
        'XGBoost': xgb,
        'LightGBM': lgbm,
        'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                       random_seed=SEED, verbose=0),
        'Stacking': StackingClassifier(
            estimators=[('rf', clone(rf)), ('xgb', clone(xgb)), ('lgbm', clone(lgbm))],
            final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1),
    }


def fit_model(model, X, y, num, cat=CAT):
    pipe = Pipeline([('prep', make_prep(num, cat)), ('model', clone(model))])
    return pipe.fit(X[num + cat], y)


def evaluate(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, pred),
        'Precision': precision_score(y_true, pred, zero_division=0),
        'Recall': recall_score(y_true, pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) else np.nan,
        'F1': f1_score(y_true, pred, zero_division=0),
        'Kappa': cohen_kappa_score(y_true, pred),
        'AUC': roc_auc_score(y_true, prob),
        'AUPRC': average_precision_score(y_true, prob),
    }


def leave_one_event_out(df, models, num, cat=CAT, target='label'):
    """Train on all events except one, test on the held-out event."""
    rows = []
    for test_event in df['event'].unique():
        tr, te = df[df['event'] != test_event], df[df['event'] == test_event]
        print(f'\nTest event: {test_event}  (train {len(tr)}, test {len(te)})')
        for name, model in models.items():
            pipe = fit_model(model, tr, tr[target], num, cat)
            p = pipe.predict_proba(te[num + cat])[:, 1]
            rows.append({'Test event': test_event, 'Model': name, **evaluate(te[target], p)})
            print(f'  {name:9s} AUC = {rows[-1]["AUC"]:.3f}')
    return pd.DataFrame(rows)
