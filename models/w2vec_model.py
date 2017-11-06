import gensim.models.word2vec as w2v
import multiprocessing
import numpy as np

FILENAME = 'model.w2v'

class Word2VecModel(object):
    
    def __init__(self):
        self._model = None
        self.filename = FILENAME
        
    @classmethod
    def from_file(cls, filename):
        self = cls()
        self._model = w2v.Word2Vec.load(filename)
        self.filename = filename
        return self
        
    def create_w2v_model(self, sentences, num_features, min_word, context,
                         sg, downsampling, seed):
        '''
        Build the word2vec model.
        :param sentences: sentence corpus.
        :param num_features: number of features length
        :param min_word: mininum word
        :param context: context size
        :param sg: whether to use skip gram 1 or cbow 0
        :param downsampling: downsampling value
        :param seed: seed
        :param directory: directory to save the model to
        :param model_name: model name
        :return: the word2vec model
        '''
    
        # define parameters
        num_features = num_features
        min_word_count = min_word
        num_workers = multiprocessing.cpu_count()
        context = context
        downsampling = downsampling
    
        # create model
        model = w2v.Word2Vec(sg=sg, seed=seed,  workers=num_workers, \
                             size=num_features, min_count=min_word_count, \
                             window=context, sample=downsampling )
    
        model.build_vocab(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.iter )
        self._model = model
        model.save(self.filename)

    def get_w2v_vector(self, sentence, vector_size):
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
            word_vector = self._model[word]
            sentence2vec[i] = word_vector
    
        return sentence2vec

    def get_w2v_vectors(self, dataframe, vector_size, colnames):
        '''
    
        :param dataset:
        :param model:
        :param vector_size:
        :param colnames:
        :param directory:
        :param name:
        :return:
        '''
        for col in colnames:
            dataframe[col] = dataframe[col].apply(lambda x: self.get_w2v_vector(x, vector_size))
    
        return dataframe
