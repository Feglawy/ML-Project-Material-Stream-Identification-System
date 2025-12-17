# Model learning 

## 1. Augumentation

first after downloading the dataset we need to augument the data to enhance the diversity This helps improve model performance and reduce overfitting by providing more varied examples for the model to learn from

by runing `src/augument.ipynb`
## 2. Feature Extraction

after getting the augumented dataset we need to extract the features out of the dataset
by running `src/feature_extraction.ipynb` this will extract a the features and store them in a file on the drive for future uses

## 3. Model learning 
### SVM
training the svm model by running  `src/SVM_model_training.ipynb` this should use the features extracted the step before 

### KNN