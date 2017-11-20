from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, merge, Reshape, Dropout
from keras.layers.merge import Dot
from keras.constraints import max_norm

FILEPATH = 'models/model.weights.best.hdf5'


class CnnModel(object):
    ''' Convolutional neural network model for question pair similarity '''
    
    def __init__(self, filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, input_shape, bias):
        # initialize the model
        self.filepath = FILEPATH
        self.cnn_model = self._build_model(filter_size, strides, padding, embedding_len, activation,filters, k_initialization,
                                           b_initialization, input_shape, bias)

    @classmethod
    def from_weights(cls, filepath, filter_size, strides, padding, embedding_len, activation,filters, k_initialization,
                                           b_initialization, input_shape, bias):
        
        self = cls(filter_size, strides, padding, embedding_len, activation, filters,
                                           k_initialization, b_initialization, input_shape, bias)
        self.filepath = filepath
        self.cnn_model.load_weights(filepath)
        return self
        

    def _convolutional_layer(self, filter_size, strides, padding, \
                            embedding_len, activation, \
                            bias, k_initialization, b_initialization, filters ):
        '''
        Defines a list of 2D convolution operations.
        :param filter_size: Integer, the dimensionality of the output space.
        :param strides: An integer or tuple/list of 2 integers, specifying the stride of the convolution.
        :param padding: Type of padding, one of  'valid' or 'same'.
        :param embedding_len: An integer, specifying the width of the 2D convolution window.
        :param activation: Activation function to use.
        :param bias: Boolean, whether the layer uses a bias vector.
        :param k_initialization: Initializer for the kernel weight matrix.
        :param b_initialization: Initializer for the bias vector.
        :param filters: A list of integer, specifying the different heights of the 2D convolution window.
        :return: A list of 2D convolution operations.
        '''
    
        cnns = [Conv2D(filters=filter_size, kernel_size=(filter_len, embedding_len), \
                       strides=strides, padding=padding, activation=activation,
                       use_bias=bias, kernel_initializer=k_initialization,
                       bias_initializer=b_initialization, kernel_constraint=max_norm(4.)) for filter_len in filters]
    
        return cnns
    

    def _input_sentence(self, shape):
        '''
        Defines the input shape.
        :param shape: Tuple of input shape
        :return: A tensor with shape (None, shape)
        '''
        return Input(shape=shape)
    
    def _cnn_sentences_layer(self, cnns, sentence):
        '''
        Computes a list of 2D convolution operations on an input sentence.
        :param cnns: a list of 2D convolution operations.
        :param sentence: input sentence.
        :return: A list of 2D convolution layer.
        '''
        return [cnn(sentence) for cnn in cnns]
    
    
    def _max_pool_sentences_layer(self, models, sentence_len, filters):
       '''
       Computes 2D max pooling operation.
       :param models: List of input tensors.
       :param sentence_len: Integer, factor by which to downscale horizontally.
       :param filters: List of factors by which to downscale vertically.
       :return: A list of tensor from the 2D max pooling operation.
       '''
       return [MaxPooling2D(pool_size=(sentence_len - filter_len + 1, 1))(model) for model, filter_len in zip(models, filters)]
    
    def _merge_concat_layer(self, model):
        '''
        Concatenates a list of tensors.
        :param model: Tensors to concatenate
        :return: A tensor from the concatenate operation.
        '''
        return  merge(model, mode='concat')
    
    def _merge_cosim_layer(self, model_1, model_2):
        '''
        Computes the cosine similarity between two tensors.
        :param model_1: The first tensor.
        :param model_2: The second tensor.
        :return: The cosine similarity value between model_1 and model_2.
        '''
        return Dot(axes=1, normalize=True)([model_1, model_2])
    
    def  _build_model(self,filter_size, strides, padding, embedding_len, activation,filters, k_initialization, b_initialization, input_shape, bias, dropout=0.3):
        '''
        Defines the convolutional neural network model.
        :param filter_size: Number of output.
        :param strides: Stride.
        :param padding: Padding value.
        :param embedding_len: Filter width.
        :param activation: Activation function.
        :param filters: List of integer - filters heights.
        :param k_initialization: Kernel initialization value.
        :param b_initialization: Bias initialization values
        :param input_shape: Input shape
        :param bias: Boolean, whether to use bias.
        :param dropout: Dropout value.
        :return: Convolutional neural network model.
        '''
        sentence_len = input_shape[0]
        
        # define input
        sentence_1_input = self._input_sentence(input_shape)
        sentence_2_input =self._input_sentence(input_shape)
    
        # convolutional layer
        cnns = self._convolutional_layer(filter_size, strides, padding, \
                            embedding_len, activation, \
                            bias, k_initialization, b_initialization, filters )
    
    
        ## sentence 1 convolutional layer
        sentence_1_cnn_layer = self._cnn_sentences_layer(cnns, sentence_1_input)
        ## add dropout regularization parameter
        sentence_1_cnn_layer = [Dropout(dropout)(cnn) for cnn in sentence_1_cnn_layer]
    
        ##sentence 2 convolutional layer
        sentence_2_cnn_layer = self._cnn_sentences_layer(cnns, sentence_2_input)
        ## add dropout regularization parameter
        sentence_2_cnn_layer = [Dropout(dropout)(cnn) for cnn in sentence_2_cnn_layer]
    
        # Max pooling layer
        ## sentence 1 max pooling layer
        sentence_1_max_pool = self._max_pool_sentences_layer(sentence_1_cnn_layer, sentence_len, filters)
    
        ## Sentence 2 max pooling layer
        sentence_2_max_pool = self._max_pool_sentences_layer(sentence_2_cnn_layer, sentence_len, filters)
    
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
        Configures the model for training.
        :param loss:  String (name of objective function) or objective function.
        :param optimizer: String (name of optimizer) or optimizer instance.
        :param metrics: list of metrics to be evaluated by the model during training and testing.
        '''
        self.cnn_model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)

    def train(self, X_train, y_train, batch_size, epochs, validation_data, verbose=2, shuffle=True ):
        '''
        Trains the model for a fixed number of epochs.
        :param X_train: List of Numpy arrays of training data.
        :param y_train: List of Numpy arrays of target data.
        :param batch_size: Number of samples per gradient update.
        :param epochs: Number of epochs to train the model.
        :param validation_data: Tuple on which to evaluate the loss and any model metric at the end of each epoch.
        :param verbose: Verbosity mode - 0, 1, 2.
        :param shuffle: Boolean (True or False)- whether to shuffle the training data before each epoch.
        '''
        checkpointer = ModelCheckpoint(filepath=self.filepath, verbose=1,
                                   save_best_only=True)
        
        self.cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                         validation_data=validation_data,
                         callbacks=[checkpointer],
                         verbose=verbose, shuffle=shuffle)
        
    def evaluate(self, X_test, y_test, verbose=0):
        '''
        Returns the loss value and metrics values for the model in test mode.
        :param X_test: List of Numpy array of test data.
        :param y_test: List of Numpy array of target data.
        :param verbose: Verbosity mode 0 or 1.
        :return: List of scalar - test loss and metrics values.
        '''
        return self.cnn_model.evaluate(X_test, y_test, verbose=verbose)
        
        
    def predict(self, x):
        '''
        Generates output predictions for the input samples.
        :param x: List of Numpy array of the input data.
        :return: Numpy array of predictions.
        '''
        return self.cnn_model.predict(x)
    
    def summary(self):
        '''
        Prints the summary representation of the model.
        '''
        self.cnn_model.summary()
