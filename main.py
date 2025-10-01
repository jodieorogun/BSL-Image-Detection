import numpy as np
import mediapipe as mp
import cv2

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize 

mpHands = mp.solutions.hands

def bothHandsCrop(image):
    rbg = np.array(image.convert("RGB"))
    with mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(rbg)
        if not results.multi_hand_landmarks:
            return None
        h, w, _ = rbg.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for handLms in results.multi_hand_landmarks:
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        cropped = rbg[y_min:y_max, x_min:x_max]
        return Image.fromarray(cropped)


def loadDataset(debug=False):
    root = Path("data")
    X, Y = [], []
    for folder in root.iterdir():
        for pic in folder.glob("*.png"):
            arr = reshapeImageWithCrop(pic)
            if arr is None:
                continue
            X.append(arr)
            Y.append(folder.name)
            if debug:
                img = Image.open(pic)
                crop = bothHandsCrop(img)
                if crop is not None:
                    outDir = Path("debug") / folder.name
                    outDir.mkdir(parents=True, exist_ok=True)
                    crop.save(outDir / f"{pic.stem}.png")
    X = np.vstack(X).astype(np.float32)
    return X, Y


def train(A, B):
    global accuracy_value
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    le = LabelEncoder()
    Y_train_enc = le.fit_transform(Y_train)
    Y_test_enc = le.transform(Y_test)
    model = KNeighborsClassifier(n_neighbors=3, metric='cosine', weights='distance')
    model.fit(X_train, Y_train_enc)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test_enc, Y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    accuracy_value = accuracy
    return model, le

def reshapeImageWithCrop(imagePath):
    image = Image.open(imagePath)
    crop = bothHandsCrop(image)
    if crop is None:
        return None
    crop = crop.resize((128, 128))
    crop = crop.convert("L")
    arr = np.asarray(crop, dtype=np.float32).flatten()
    arr = arr.reshape(1, -1)
    return normalize(arr, norm='l2')

def reshapeFrameWithCrop(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    crop = bothHandsCrop(image) 
    if crop is None:
        return None
    crop = crop.resize((128, 128))
    crop = crop.convert("L")
    arr = np.asarray(crop, dtype=np.float32).flatten().reshape(1, -1)
    return normalize(arr, norm='l2')

if __name__ == "__main__":
    X, Y = loadDataset()
    model, le = train(X, Y)
    with open("accuracy.txt", "w") as f:
        f.write(f"accuracy: {accuracy_value * 100} \n")  
    pred = model.predict(reshapeImageWithCrop("my_predictionB.png"))
    predLabel = le.inverse_transform(pred)
    print(f"Predicted label: {predLabel}")

    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Could not open webcam")
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            feat = reshapeFrameWithCrop(frame)
            if feat is not None:
                pred = model.predict(feat)
                label = le.inverse_transform(pred)[0]
                text = f"Pred: {label}"
                color = (0, 255, 0)
            else:
                text = "no hands"
                color = (0, 0, 255)

            # overlay text
            cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # show the live feed
            cv2.imshow("BSL Live", frame)

            # quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()