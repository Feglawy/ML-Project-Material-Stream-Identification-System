# ML Project: Material Stream Identification System


## project setup
1. create virtual enviroment

```shell
python -m venv .venv
```

2. activate the vertual enviroment
```shell
.venv\scripts\activate
```

3. install required libraries
```shell
pip install -r .\requirements.txt
```

## How to run 

1. First download the dataset 
the data isn't balanced so we need to augument the data and balance it to imporve performance and reduce overfitting

2. Augumentation

by running `augument.ipynb`
this step we will make a seperate dataset called for example `aug_dataset`

3. Feature Extraction

by running `feature_extraction_torch.ipynb`
this well 

4. Model learning

after the previous step you should have a `feature` directory this will contain the extracted features
it will be load by the model to learn from

run the [KNN](src/KNN_model_training.ipynb) or [SVM](src/SVM_model_training.ipynb)

this will extract the model and save it to `models` directory

"FOR THOSE WHO COME AFTER".

5. System deployment

after the models extracted 

run the [Camera app](src/camera_app.ipynb) this will run an seperate window with the camera 
and the detection will be in outputed in the cli


## Dataset 
[Dataset URL](https://drive.google.com/file/d/1IKsJglhHUSyaEmQAGVUHo_4FTu_ya9P6/view?usp=sharing)

The dataset contains the following
```
cardboard: 259
glass: 401
metal: 328
paper: 476
plastic: 386
trash: 110
```