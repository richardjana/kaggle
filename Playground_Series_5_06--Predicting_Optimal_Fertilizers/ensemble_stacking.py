import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.append('/'.join(__file__.split('/')[:-2]))
from kaggle_utilities import mapk


NUM_CLASSES = 7

# 1) Load the saved predictions
file_paths = [
    'xgb_stacking_data.pkl',
    'lgbm_stacking_data.pkl',
    'catboost_stacking_data.pkl'
]

oof_preds_list = []
test_preds_list = []

for path in file_paths:
    data = joblib.load(path)
    oof_preds_list.append(data['oof_preds'])
    test_preds_list.append(data['test_preds'])
    y_train = data['y_train']


# 2) Stack them horizontally
X_meta_train = np.hstack(oof_preds_list)
X_meta_test = np.hstack(test_preds_list)


# 3) Train a meta-classifier
meta_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
meta_model.fit(X_meta_train, y_train)

meta_preds = meta_model.predict_proba(X_meta_test)

map_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X_meta_train, y_train):
    X_tr, X_val = X_meta_train[train_idx], X_meta_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    meta_model.fit(X_tr, y_tr)
    val_preds = meta_model.predict_proba(X_val)
    top_3 = np.argsort(-val_preds, axis=1)[:, :3]
    actual = [[label] for label in y_val]
    map_scores.append(mapk(actual, top_3, k=3))

print(f"Mean MAP@3: {np.mean(map_scores):.5f}")

# 4) Coefficients from a Linear Meta-Model plotting
coefs = meta_model.coef_
n_base_models = len(coefs[0]) // NUM_CLASSES  # if stacking probs per class

# Sum importance across classes
importance = np.sum(coefs, axis=0).reshape(n_base_models, -1).mean(axis=1)

plt.bar(range(n_base_models), importance)
plt.xlabel('Base Model')
plt.ylabel('Average Weight')
plt.title('Base Model Contributions to Meta Model')
plt.xticks(range(n_base_models), ['XGB', 'CatBoost', 'LGBM'])  # or custom labels
plt.show()
