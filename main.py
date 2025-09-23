import numpy as np
import mediapipe as mp

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def loadDataset():
    root = Path("data")
    X, Y = [], []
    for folder in root.iterdir():
        for pic in folder.glob("*png"):
            image = Image.open(pic)
            image = image.resize((128, 128))
            image = image.convert("L")
            arr = np.asarray(image, dtype=np.float32).flatten()
            X.append(arr)
            Y.append(folder.name)
    return X, Y


def train(A, B):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    le = LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Y_test_enc = le.transform(Y_test)
    model = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
    model.fit(X_train, Y_train_enc)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test_enc, Y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return model, le

def reshape(imagePath):
    image = Image.open(imagePath)
    image = image.resize((128, 128))
    image = image.convert("L")
    arr = np.asarray(image, dtype=np.float32).flatten()
    return arr.reshape(1, -1)

if __name__ == "__main__":
    X, Y = loadDataset()
    model, le = train(X, Y)
    pred = model.predict(reshape("my_prediction.png"))
    predLabel = le.inverse_transform(pred)
    print(f"Predicted label: {predLabel[0]}")