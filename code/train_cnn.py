from code.cnn_model import CnnModel
from code.utils import *
import pandas as pd

filter_size = 32
strides = (2, 2)
padding = 'SAME'
embedding_len = 100
activation = 'relu'
filters = [2, 3, 4, 5]
bias = True
k_initialization = 'glorot_uniform'
b_initialization = 'zeros'
input_shape = (20, 100, 1)
pool_size = 2

seed = 10
FILE_PATH = '/data/quora_cnn.pickle'
def main():
    # # prepare data
    data_cnn = pd.read_pickle(FILE_PATH)
    # print(data_cnn.head())
    # print(data_cnn.shape)

    train, validation, test = split_dataset(data_cnn, seed, [0.8, 0.9])

    # # training data
    sentence_1_train = train.question1.tolist()
    sentence_2_train = train.question2.tolist()
    y_train = train.is_duplicate.tolist()
    question_1_train, question_2_train, y_train = reshape_data(sentence_1_train, sentence_2_train, y_train)

    # print('Training data')
    # print(f'Question 1 training data size {question_1_train.shape}')
    # print(f'Question 2 training data size {question_2_train.shape}')
    # print(f'Y training data size {y_train.shape}')
    # print()
    #
    # # validation data
    sentence_1_validation = validation.question1.tolist()
    sentence_2_validation = validation.question2.tolist()
    y_validation = validation.is_duplicate.tolist()
    question_1_validation, question_2_validation, y_validation = reshape_data(sentence_1_validation, sentence_2_validation, y_validation)

    # print('Validation data')
    # print(f'Question 1 validation data size {question_1_validation.shape}')
    # print(f'Question 2 validation data size {question_2_validation.shape}')
    # print(f'Y validation data size {y_validation.shape}')
    # print()
    #
    # # testing data
    sentence_1_test = test.question1.tolist()
    sentence_2_test = test.question2.tolist()
    y_test = test.is_duplicate.tolist()
    question_1_test, question_2_test, y_test = reshape_data(sentence_1_test, sentence_2_test, y_test)
    # print('Test data')
    # print(f'Question 1 test data size {question_1_test.shape}')
    # print(f'Question 2 test data size {question_2_test.shape}')
    # print(f'Y test data size {y_test.shape}')
    # print()

    cnn_model = CnnModel(filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, pool_size, input_shape, bias)
    cnn_model.summary()
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fbeta])

    cnn_model.train([question_1_train, question_2_train], y_train, batch_size=256, epochs=1000,
                  validation_data=([question_1_validation, question_2_validation], y_validation),
                  verbose=2, shuffle=True)

    # cnn_model = CnnModel.from_weights(filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, pool_size, input_shape, bias)
    # cnn_model.summary()
    # cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    scores = cnn_model.evaluate([question_1_test, question_2_test], y_test)
    
    print("CNN Accuracy Error: %.2f%%" % (100 - scores[1] * 100))
    # pred = cnn_model.predict([question_1_test, question_2_test])
    # for elt in range(len(pred)):
    #     print(pred[elt], y_test[elt])
    print("CNN fbeta Error: %.2f%%" % (100 - scores[2] * 100))

if __name__ == '__main__':
    main()
    
    
