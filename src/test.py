import os
import cv2 as cv
import numpy as np
import joblib
import pandas as pd

import torch
from torchvision import models, transforms

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def feature_extraction(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = resnet(x)
    return f.view(-1).cpu().numpy()   # (2048,)


def _predict(model, feature, threshold):
    if threshold == None:
        threshold = 0.6

    #  SVM
    if isinstance(model, SVC):
        probs = model.predict_proba(feature)[0]
        if np.max(probs) < threshold:
            return 6
        return int(np.argmax(probs))

    #  kNN
    elif isinstance(model, KNeighborsClassifier):
        distances, _ = model.kneighbors(feature)
        if distances.mean() > threshold:
            return 6
        return int(model.predict(feature)[0])
    
    else:
        return 6


def predict(dataFilePath, modelPath):
    predictions = []

    modelPath = "./models/knn_model.pkl"
    data_loaded = joblib.load(modelPath)
    model = data_loaded["model"]
    scaler: StandardScaler = data_loaded["scaler"]
    rejection_threshold = data_loaded["rejection_threshold"]

    img_files = os.listdir(dataFilePath)

    for img_name in img_files:
        img_path = os.path.join(dataFilePath, img_name)
        
        img = cv.imread(img_path)
        if img is None:
            predictions.append(6)
            continue
        
        feature = feature_extraction(img)
        feature = feature.reshape(1, -1) 
        feature = scaler.transform(feature)

        pred = _predict(model, feature, rejection_threshold)

        predictions.append(pred)
    return predictions



def verify_sample(sample_dir_path):
    y_labals = {
        0: "glass",
        1: "paper",
        2: "cardboard",
        3: "plastic",
        4: "metal",
        5: "trash",
        6: "unknown"
    }
    
    names = os.listdir(sample_dir_path)
    
    output = {
        "ImageName": names,
        "predictedlabel": []
    }

    predictions = predict(sample_dir_path, "./models/svm_model.pkl")

    for i in range(len(predictions)):
        predictions[i] = y_labals[predictions[i]]

    output["predictedlabel"] = predictions

    df = pd.DataFrame(output)
    print(df)
    df.to_excel(f"./output.xlsx", index=False)



if __name__ == "__main__":
    verify_sample("./sample")