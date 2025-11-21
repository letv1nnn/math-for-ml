import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


train_dataset_file = "datasets/spaceship-titanic/train.csv"
test_dataset_file = "datasets/spaceship-titanic/test.csv"
train_dataset = pd.read_csv(train_dataset_file)
x_test = pd.read_csv(test_dataset_file)
passengers = x_test["PassengerId"]

# Splitting the data into train set and cross validation set.
x_train, x_cv, y_train, y_cv = train_test_split(
    train_dataset.iloc[:, :13], train_dataset["Transported"],
    random_state=42, test_size=0.2
)


# Feature Engineering
# I'm dropping these columns "PassengerId", "Name", "HomePlanet" and all services,
# because they do not influence on the result

# Need to fill: Cabin, Destination, Age
def feature_engineering(x):
    x = x.copy()
    x = x.drop(columns=[
        "PassengerId", "Name", "HomePlanet", "RoomService",
        "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ])
    # Change all boolean representation into binary
    x["CryoSleep"] = x["CryoSleep"].apply(lambda item: 0.0 if item is False else 1.0)
    x["VIP"] = x["VIP"].apply(lambda item: 0.0 if item is False else 1.0)
    # Converting all destination values into a number representation and filling nan values
    le = LabelEncoder()
    x["Destination"] = le.fit_transform(x["Destination"])
    # Hande the cabin case with its deck/num/side, so I'll add one new column.
    x["CabinNumber"] = x["Cabin"].apply(
        lambda item: 0.0 if pd.isna(item)
        else float(item.split("/")[1])
    )
    x["Cabin"] = x["Cabin"].apply(
        lambda item: 0.0 if pd.isna(item)
        else 1.0 if item.split("/")[2] == "S"
        else 2.0
    )
    # Handling nan cases in Age column
    average_age = x["Age"].mean()
    x["Age"] = x["Age"].apply(
        lambda item: average_age if pd.isna(item)
        else item
    )

    return x

# Apply feature engineering function to all datasets
x_train, x_cv, x_test = feature_engineering(x_train), feature_engineering(x_cv), feature_engineering(x_test)
print(x_train.info())


# Scaling all data
scaler = MinMaxScaler()
x_train, x_cv, x_test = scaler.fit_transform(x_train), scaler.transform(x_cv), scaler.transform(x_test)


# Building a model
model = HistGradientBoostingClassifier(random_state=42)
model.fit(x_train, y_train)


# Testing the model in a training and dev sets
y_pred_train = model.predict(x_train)
y_pred_cv = model.predict(x_cv)
accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training set accuracy: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_cv, y_pred_cv)
print(f"Cross Validation set accuracy: {accuracy * 100:.2f}%")


# Visualization
def visualize():
    train_dataset.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.tight_layout()
    plt.show()

#visualize()


# Final test prediction
result_prediction = model.predict(x_test)
result_df = pd.DataFrame({
    "PassengerId": passengers,
    "Transported": result_prediction
})

# Save the result
file = "datasets/submission.csv"
result_df.to_csv(file, index=False)
