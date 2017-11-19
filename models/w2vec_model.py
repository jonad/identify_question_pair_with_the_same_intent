import gensim.models.word2vec as w2v
import multiprocessing
import numpy as np

FILENAME = 'model.w2v'

class Word2VecModel(object):
    '''In domain word embeddings model using word2vec algorithm'''
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
        Builds the word2vec model from a corpus of text.
        :param sentences: List of list of string, the corpus.
        :param num_features: Integer, embedding length.
        :param min_word: Integer, the algorithm will ignore all word with frequency lower than min_word.
        :param context: Integer, window size, the maximum distance between the current and predicted word within the sentence.
        :param sg: Integer - 0 or 1 -, which indicates whether to use the skip-gram archittecture (1), or the continuous bag of words archittecture (0).
        :param downsampling: Threshold for configuring which higher-frequency words are randomly downsampled;
        :param seed: Integer, a random seed.
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
        Gets the word representations of a given sentence.
        :param sentence: List of string, the input sentence.
        :param vector_size: Integer, embedding size.
        :return: A list of word representations.
        '''
    
        sentence_len = len(sentence)
        sentence2vec = np.zeros(shape=(sentence_len, vector_size), dtype='float32')
        for i in range(sentence_len):
            word = sentence[i]
            word_vector = self._model[word]
            sentence2vec[i] = word_vector
    
        return sentence2vec
