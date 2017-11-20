import pandas as pd

from models.cnn_model import *
from utils.utils import *
from sklearn.model_selection import train_test_split

# define parameters
filter_size = 32
strides = (1, 1)
padding = 'VALID'
embedding_len = 100
activation = 'relu'
filters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
bias = True
sentence_len = 40
k_initialization = 'glorot_uniform'
b_initialization = 'zeros'
input_shape = (sentence_len, embedding_len, 1)
batch_size = 32
epochs = 100
seed = 10

# define metrics
acc = make_accuracy(0)
fbeta = make_fbeta(0)

FILE_PATH = 'data/test_cnn.pickle'
def main():
    # Prepare data
    data_cnn = pd.read_pickle(FILE_PATH)
    train, validation = train_test_split(data_cnn, test_size=0.1, random_state=seed)
    
    ## Traning data
    sentence_1_train = train.question1.tolist()
    sentence_2_train = train.question2.tolist()
    y_train = train.is_duplicate.tolist()
    question_1_train, question_2_train, y_train = reshape_data(sentence_1_train, sentence_2_train, y_train)
    
    ## Validation data
    sentence_1_validation = validation.question1.tolist()
    sentence_2_validation = validation.question2.tolist()
    y_validation = validation.is_duplicate.tolist()
    question_1_validation, question_2_validation, y_validation = reshape_data(sentence_1_validation,
                                                                              sentence_2_validation, y_validation)

    # Define the model
    cnn_model = CnnModel(filter_size, strides, padding, embedding_len, activation, filters, k_initialization,
                         b_initialization, input_shape, bias)
    cnn_model.summary()
    cnn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=[acc, fbeta])
    
    # Train the model
    cnn_model.train([question_1_train, question_2_train], y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=([question_1_validation, question_2_validation], y_validation),
                    verbose=2, shuffle=True)
    


if __name__ == '__main__':
    main()
    
    
