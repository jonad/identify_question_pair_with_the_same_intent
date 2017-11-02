from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, merge, Reshape
from keras.layers.merge import Dot

FILEPATH = '/output/model.weights.best.hdf5'


class CnnModel(object):
    def __init__(self, filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, pool_size, input_shape, bias):
        # initialize the model
        self.filepath = FILEPATH
        self.cnn_model = self._build_model(filter_size, strides, padding, embedding_len, activation,filters, k_initialization,
                                           b_initialization, pool_size, input_shape, bias)

    @classmethod
    def from_weights(cls, filepath, filter_size, strides, padding, embedding_len, activation,filters, k_initialization,
                                           b_initialization, pool_size, input_shape, bias):
        
        self = cls(filter_size, strides, padding, embedding_len, activation, filters,
                                           k_initialization, b_initialization, pool_size, input_shape, bias)
        self.filepath = filepath
        self.cnn_model.load_weights(filepath)
        return self
        

    def _convolutional_layer(self, filter_size, strides, padding, \
                            embedding_len, activation, \
                            bias, k_initialization, b_initialization, filters ):
        '''
        Define a list of convolutions.
        :param filter_size: filter size value
        :param strides: strides value
        :param padding: padding
        :param embedding_len: embedding length
        :param activation: activation function
        :param bias: whether to use bias or not
        :param k_initialization: kernel initialization value
        :param b_initialization: bias initialization values
        :param filters: filter length list
        :return: a list of convolution.
        '''
    
        cnns = [Conv2D(filters=filter_size, kernel_size=(filter_len, embedding_len), \
                       strides=strides, padding=padding, activation=activation,
                       use_bias=bias, kernel_initializer=k_initialization,
                       bias_initializer=b_initialization) for filter_len in filters]
    
        return cnns
    
    


    def _input_sentence(self, shape):
        '''
        Define the input tensor
        :param shape: input shape
        :return: a tensor with the given shape
        '''
        return Input(shape=shape)
    
    def _cnn_sentences_layer(self, cnns, sentence):
        '''
        Compute convolutional tensor with different filter length
        :param cnns: list of tensor with different filter
        :param input_shape: input tensor
        :return: a list of convolutional tensor with different filter length.
        '''
        return [cnn(sentence) for cnn in cnns]
    
    
    def _max_pool_sentences_layer(self, models, pool_size):
        '''
        perform a max pooling of a given tensor
        :param models: tensor to perform a max pooling
        :param pool_size: pooling size
        :return: a tensor
        '''
        return [MaxPooling2D(pool_size=pool_size)(model) for model in models]
    
    def _merge_concat_layer(self, model):
        '''
        Concatenate filter of a given tensor
        :param model: tensor to concatenate
        :return: concatenated tensor
        '''
        return  merge(model, mode='concat')
    
    def _merge_cosim_layer(self, model_1, model_2):
        '''
        cosine similarity layer
        :param model_1: model 1
        :param model_2: model 2
        :return: a tensor which is a cosine similarity value between model_1 and model_2
        '''
        return Dot(axes=1, normalize=True)([model_1, model_2])
    
    def  _build_model(self,filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, pool_size, input_shape, bias):
        '''
        create a Convolutional neural network
        :param filter_size: filter size value
        :param strides: stride tuple
        :param padding: padding value
        :param embedding_len: length of embedding sentences
        :param activation: activation function.
        :param filters: filter array
        :param k_initialization: kernel initialization value.
        :param b_initialization: bias initialization values
        :param pool_size: value or tuple for the pool size
        :param input_shape: input shape
        :param bias: boolean value for whether or not to use bias.
        :return: Convolutional neural network model.
        '''
        # define input
        sentence_1_input = self._input_sentence(input_shape)
        sentence_2_input =self._input_sentence(input_shape)
    
        # cnn layer
        cnns = self._convolutional_layer(filter_size, strides, padding, \
                            embedding_len, activation, \
                            bias, k_initialization, b_initialization, filters )
    
    
        ## sentence 1 cnn layer
        sentence_1_cnn_layer = self._cnn_sentences_layer(cnns, sentence_1_input)
    
        ##sentence 2 cnn layer
        sentence_2_cnn_layer = self._cnn_sentences_layer(cnns, sentence_2_input)
    
        # Max pool layer
        ## sentence 1 max pool
        sentence_1_max_pool = self._max_pool_sentences_layer(sentence_1_cnn_layer, pool_size)
    
        ## Sentence 2 max pool
        sentence_2_max_pool = self._max_pool_sentences_layer(sentence_2_cnn_layer, pool_size)
    
        # concat layer
        ## Sentence 1 concat layer
        sentence_1_concat =  self._merge_concat_layer(sentence_1_max_pool)
    
        ## sentence 2 concat layer
        sentence_2_concat = self._merge_concat_layer(sentence_2_max_pool)
    
        # Flatten layer
        ## sentence 1 flatten layer
        sentence_1_flatten = Reshape((-1, ))(sentence_1_concat)
    
        ## Sentence 2 Flatten layer
        sentence_2_flatten = Reshape((-1, ))(sentence_2_concat)
    
        # Merge with cosine similarity layer
        dot = self._merge_cosim_layer(sentence_1_flatten, sentence_2_flatten)
        model = Model([sentence_1_input, sentence_2_input], [dot])
    
        return model
    
    def compile(self, loss, optimizer, metrics ):
        '''
        
        :param loss:
        :param optimizer:
        :param metrics:
        :return:
        '''
        self.cnn_model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)

    def train(self, X_train, y_train, batch_size, epochs, validation_data, verbose=2, shuffle=True ):
        '''
        Train the model
        :param X_train:
        :param y_train:
        :param batch_size:
        :param epochs:
        :param validation_data:
        :param verbose:
        :param shuffle:
        :return:
        '''
        checkpointer = ModelCheckpoint(filepath=self.filepath, verbose=1,
                                   save_best_only=True)
        self.cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                         validation_data=validation_data,
                         callbacks=[checkpointer],
                         verbose=verbose, shuffle=shuffle)
        
    def evaluate(self, X_test, y_test, verbose=0):
        '''
        Evaluate the model
        :param X_test:
        :param y_test:
        :param verbose:
        :return:
        '''
        return self.cnn_model.evaluate(X_test, y_test, verbose=verbose)
        
        
    def predict(self, x):
        '''
        Predict x
        :param x:
        :return:
        '''
        return self.cnn_model.predict(x)
    
    def summary(self):
        '''
        Summarize the model.
        :return: output the summary data.
        '''
        self.cnn_model.summary()
    
