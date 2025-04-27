import shap
import matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
import pickle

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data_stress.csv")
data.dropna(inplace=True)

X = data.drop(labels = ["Stress Levels"],axis = 1)
y = data["Stress Levels"].values

feature_names = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X100 = shap.utils.sample(X_train, 100)
print(X100.columns.tolist())

param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    {'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
]

lr = LogisticRegression(random_state = 101, max_iter=5000)
cv = RepeatedKFold(n_splits = 10, n_repeats = 10, random_state = 101)
grid_cv = GridSearchCV(lr, param_grid, cv = cv, n_jobs = -1)
grid_cv.fit(X_train, y_train)
y_predict = grid_cv.best_estimator_.predict(X_test)

print("Accuracy: ", grid_cv.best_estimator_.score(X_test, y_test))

explainer = shap.Explainer(grid_cv.best_estimator_.predict, X100)
shap_values = explainer(X100)

shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.scatter(shap_values)

def model_fn(X):
    return grid_cv.best_estimator_.predict_proba(X)[:, 1]

shap.partial_dependence_plot(
    "heart rate ",
    model_fn,
    X100,
    ice=False,
    model_expected_value=True,
    feature_expected_value=True,
    show=True,
)

with open('stress_detection_model.pkl','wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(grid_cv, f)

with open('stress_scaler_model.pkl', 'wb') as f:
    pickle.dump(scaler, f)