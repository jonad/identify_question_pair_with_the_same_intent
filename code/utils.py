import numpy as np
import keras.backend as K
import string
#from nltk.corpus import stopwords
import re
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import csv

#stop_words = stopwords.words('english')

def split_dataset(dataset, seed, indices):
    '''
    Split the dataset into train, test, and validation
    :param dataset: the dataset
    :param seed: seed
    :param indices: indices array
    :return: return tuple of trai, test and validation data.
    '''
    train, validation, test = np.split(dataset.sample(frac=1, random_state=seed),
                                     [int(indices[0]* len(dataset)), int(indices[1] * len(dataset))])

    return train, validation, test

def reshape_data(sentence_1_data, sentence_2_data, y_data):
    '''
    Reshape the input data
    :param sentence_1_data: sentence 1
    :param sentence_2_data: sentence 2
    :param y_data: output
    :return: a reshaped sentence 1, sentence 2, output
    '''
    # sentence 1
    sentence_1 = np.array(sentence_1_data)
    sentence_1 = sentence_1.reshape(sentence_1.shape +  (1, ))
    
    # sentence 2
    sentence_2 = np.array(sentence_2_data)
    sentence_2 = sentence_2.reshape(sentence_2.shape + (1,))
    
    # y
    y = np.array(y_data)
    y  = y.reshape((y.shape[0], 1))
    
    return sentence_1, sentence_2, y

def accuracy(y_true, y_pred, threshold_shift=0):
    '''
    Compute accuracy score
    :param y_true: actual output
    :param y_pred: predicted output
    :param threshold_shift: threshold
    :return: accuracy score
    '''
    y_pred = K.clip(y_pred, 0, 1)
    y_pred = K.round(y_pred + threshold_shift)
    return K.mean(K.equal(y_true, y_pred))

def clean_str(txt):
    '''
    clean a sentence by removing all non alphanumeric token.
    '''
    txt = str(txt)
    txt = re.sub(r"[^A-Za-z0-9(),!?\'\`&%]", " ", txt)
    txt = re.sub(r"\'s", " \'s", txt)
    txt = re.sub(r"\'ve", " \'ve", txt)
    txt = re.sub(r"\'t", " n\'t", txt)
    txt = re.sub(r"\'re", " \'re", txt)
    txt = re.sub(r"\'d", " \'d", txt)
    txt = re.sub(r"\'ll", " \'ll", txt)
    txt = re.sub(r",", " , ", txt)
    txt = re.sub(r"!", " ! ", txt)
    txt = re.sub(r"\(", " ( ", txt)
    txt = re.sub(r"\)", " ) ", txt)
    txt = re.sub(r"\?", " ? ", txt)
    txt = re.sub(r"\&", " & ", txt)
    txt = re.sub(r"\%", " percent ", txt)
    txt = txt.strip().lower()
    return txt

def remove_punctuation(txt):
    '''
    Remove punctuation in a given text.
    :param txt: a text string
    :return: an array of word which is not a punctuation.
    '''
    return [w for w in txt if w not in string.punctuation]


def remove_stopwords(txt):
    '''
    Remove english stopword in a given text.
    :param txt: a string of text.
    :return: an array of word which is not english stopwords.
    '''
    return [w for w in txt if w not in stop_words]

def fbeta(y_true, y_pred, threshold_shift=0, beta=1):
    '''
    Compute fbeta score.
    :param y_true: y_true
    :param y_pred: predicted value
    :param threshold_shift: threshold
    :param beta: beta value
    :return: fbeta score
    '''
    
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    y_true = K.clip(y_true, 0, 1)
    
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
    
    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def get_w2v_vector(model, sentence, vector_size):
    '''
    Get the word2vec vector of a given sentence.
    :param model: word2vec model
    :param sentence: sentence
    :param vector_size: vector size
    :return: a word2vec vector of a given sentence
    '''
    
    sentence_len = len(sentence)
    sentence2vec = np.zeros(shape=(sentence_len, vector_size), dtype='float32')
    for i in range(sentence_len):
        word = sentence[i]
        word_vector = model[word]
        sentence2vec[i] = word_vector
    
    return sentence2vec


def cosine_sim(u, v):
    '''
    Compute the cosine similarity between two vectors
    :param u: a vector
    :param v: a vector
    :return: the cosine similarity score.
    '''
    return np.dot(u,v) / (norm(u)*norm(v))


def similarity(q1, q2):
    '''
    Compute the similarity score between arrays of vectors.
    :param q1: array of vectors
    :param q2: array of vectors
    :return: vector of similarity score
    '''
    sim = []
    for el1, el2 in zip(q1, q2):
        sim.append(cosine_sim(el1, el2))

    return sim

def save_result(filename, results):
    '''
    Save a dictionary of data into a file.
    :param filename: the file name
    :param results: The dictionary to save into the file
    :return: None
    '''
    keys = results[0].keys()
    with open(filename, 'w') as f:
        dict_writer = csv.DictWriter(f,  keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def split_dataset_xgb(x, y, test_size, seed):
  '''
    Split the data into train and test data.
    :param x: input data
    :param y: target data
    :param test_size: proportion of the test size
    :param seed: seed
    :return: train and test dataset
  '''
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
  return X_train, X_test, y_train, y_test
