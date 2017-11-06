# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: Identify question pair with the same intent.
### Install
* [**Python 3.6**](https://www.python.org/downloads/release/python-350/)
*  [**Gensim**](https://radimrehurek.com/gensim/install.html)
* [**Keras 2.0**](https://keras.io/)
* [**NLTK**](http://www.nltk.org/)

### Dataset
* [**Training data**](https://www.kaggle.com/c/quora-question-pairs/data)

### Step 1: preprocess data
Navigate to **notebooks** folders and run the following command

```bash
jupyter notebook data_preprocessing.ipynb
```
After running all cells in the notebooks, you will have a preprocessed
data into the data folder
### Step 2:
Train either the xgboost model or the cnn model by running the
following command from the project's root directory
```bash
python train_cnn.py
```
This will train the neural network model. You can modify this file 
to run the model with different parameters
```bash
python train_xgb.py
```
This will train the xgboost model. You can modify this file to run the model
with different parameter.
### Do not run this code on your laptop, run it on a GPU instance on the cloud.
