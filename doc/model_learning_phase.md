# Model learning 

## 1. Augumentation

first after downloading the dataset we need to augument the data to enhance the diversity This helps improve model performance and reduce overfitting by providing more varied examples for the model to learn from

by runing `src/augument.ipynb`
## 2. Feature Extraction

after getting the augumented dataset we need to extract the features out of the dataset
by running `src/feature_extraction.ipynb` this will extract a the features and store them in a file on the drive for future uses

### resnet50 over hog + hst + lbp

using resnet50 and extaacting the features made the model more accurate than manually extracting the hog + hst + lbp features 

this made the models more accurate by at least 30% percent

as the resnet extracted features scored a 94% accuracy on svm on the other hand the hog + hst + lbp scored at best a 66% accuracy

## 3. Model learning 

### SVM
training the svm model by running  `src/SVM_model_training.ipynb` this should use the features extracted the step before 

after running the model will be extracted 
and loaded again using 
```python
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data_loaded = joblib.load(model_path)
svm : SVC = data_loaded["model"]
scaler: StandardScaler = data_loaded["scaler"]
```

### KNN

training the knn model by runing `src/KNN_model_training.ipynb` this should use the features extracted

just like the svm model this will extract the best knn model with its parametars
load the model again using 
```python
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
    {
        "model": best_model,
        "scaler": scaler,
        "rejection_threshold": rejection_threshold,
        "best_params": best_params
    },
data_loaded = joblib.load(model_path)
knn : KNeighborsClassifier  = data_loaded["model"]
scaler: StandardScaler = data_loaded["scaler"]
rejection_threshold = data_loaded["rejection_threshold"]
best_params = data_loaded["best_params"]
```

