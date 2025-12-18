# %%
import cv2 as cv
import numpy as np
import torch
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ResNet50 without classifier
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




