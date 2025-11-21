import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



def preprocess_data(df: pd.DataFrame):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Embarked"] = df["Embarked"].fillna("S")
    df.drop(columns=["Embarked"], inplace=True)
    fill_missing_ages(df)
    # feature engineering
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    return df


def fill_missing_ages(df: pd.DataFrame):
    ht = {}
    for i in df["Pclass"].unique():
        if i not in ht:
            ht[i] = df[df["Pclass"] == i]["Age"].median()
    df["Age"] = df.apply(lambda row: ht[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)


dataset = pd.read_csv("archive/tested.csv")
dataset = preprocess_data(dataset)

X = dataset.drop(columns=["Survived"])
y = dataset["Survived"]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x.values.reshape(-1, X.shape[1]))


# KNN model
def tune_model(train_x, train_y):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    return grid_search.best_estimator_


best_model = tune_model(train_x, train_y)


def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix


def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sb.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.show()


accuracy, matrix = evaluate_model(best_model, test_x, test_y)
print(f"Accuracy: {accuracy * 100:.2f}%\nConfusion matrix: {matrix}")
plot_model(matrix)
