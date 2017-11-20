import numpy as np
import keras.backend as K
import string
from nltk.corpus import stopwords
import re
from numpy.linalg import norm
import csv

stop_words = stopwords.words('english')


def reshape_data(sentence_1_data, sentence_2_data, y_data):
    '''
    Reshapes the input data.
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



def clean_str(txt):
    '''
    Cleans a text by removing all non alphanumeric characters.
    :param txt: String, the input sentence.
    :return: A cleaned sentence.
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
    Removes punctuations in a given sentence.
    :param txt: String, input sentence.
    :return: a list of words from the input sentence without punctuation.
    '''
    return [w for w in txt if w not in string.punctuation]


def remove_stopwords(txt):
    '''
    Removes english stopword in a given sentence.
    :param txt: String, the input sentence.
    :return: A list of words from the input sentence without any stopwords.
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




def cosine_sim(u, v):
    '''
    Computes the cosine similarity between two vectors u and v.
    :param u: Numpy ndarray, the vector u.
    :param v: Numpy ndarray, the vector v.
    :return: Float between 0 and 1, the cosine similarity score between the vector u and v.
    '''
    return np.dot(u,v) / (norm(u)*norm(v))


def similarity(q1, q2):
    '''
    Computes the similarity score between lists of vectors.
    :param q1: List of the first vector.
    :param q2: List of the second vector.
    :return: A list of similarity score between vectors in q1 and q2.
    '''
    sim = []
    for el1, el2 in zip(q1, q2):
        sim.append(cosine_sim(el1, el2))

    return sim

def save_result(filename, results):
    '''
    Saves a dictionary of data into a file.
    :param filename: String, the filename
    :param results: The dictionary to save into the file
    '''
    keys = results[0].keys()
    with open(filename, 'w') as f:
        dict_writer = csv.DictWriter(f,  keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def make_accuracy(threshold_shift):
    '''
    Creates an accuracy function with a given threshold.
    :param threshold_shift: Float, threshold
    :return: An accuracy function
    '''
    def accuracy(y_true, y_pred):
        '''
        Compute accuracy score
        :param y_true: actual output
        :param y_pred: predicted output
        :return: accuracy score
        '''
        y_pred = K.clip(y_pred, 0, 1)
        y_pred = K.round(y_pred + threshold_shift)
        return K.mean(K.equal(y_true, y_pred))
    return accuracy
    

def make_fbeta(threshold_shift):
    '''
    Creates fbeta function with a given threshold.
    :param threshold_shift: Float, threshold shift
    :return: An fbeta function
    '''
    def fbeta(y_true, y_pred, beta=1):
        '''
        Computes the fbeta score.
        :param y_true: y_true
        :param y_pred: predicted value
        :param beta: beta value.
        :return: fbeta score.
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
    
    return fbeta
