from pathlib import Path
import numpy as np
"""
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
"""

def loadDataset():
    root = Path("data")
    X, Y = [], []
    for folder in root.iterdir():
        print("hi")
        for pic in folder:
            print("yo")

if __name__ == "__main__":
    loadDataset()