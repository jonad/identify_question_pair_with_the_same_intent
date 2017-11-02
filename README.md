# Machine Learning Engineer Nanodegree
# Capstone Project
## Project: Identify question pair with the same intent.
### Install
* [**Python 3.5**](https://www.python.org/downloads/release/python-350/)
*  [**Gensim**](https://radimrehurek.com/gensim/install.html)
* [**Keras 2.0**](https://keras.io/)
* [**NLTK**](http://www.nltk.org/)

### Dataset
* [**Training data**](https://www.kaggle.com/c/quora-question-pairs/data)

### Folders
#### Notebooks
This folder contains two important files:
* **proposal.ipynb** contains the description of the project.To run,
just navigate to the current folder and run the following command:
```bash
jupyter notebook proposal.ipynb
```
* **data_preprocessing.ipynb** contains all the data preprocessing steps.
to run just navigate to the curr folder and run the following command
```bash
jupyter notebook data_preprocessing.ipynb
```

#### Code
This folder contains two important files:
* **train_cnn.py** contains the code to build the neural network model.
To run, just navigate to the folder and run following command
```bash
python train_cnn.py
```

* **train_xgb.py** contains the code to build the xgboost model.
To run, just navigate to the folder and run the following command
```bash
python train_xgb.py
```

### Do not run this code on your laptop, run it on a GPU instance on the cloud.
