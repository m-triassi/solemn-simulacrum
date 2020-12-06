import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, GRU
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential

from processor import DataProcessor
import os
from dotenv import load_dotenv
load_dotenv()

class SimulacrumGenerator:

    def __init__(self, max_words=1000, max_len=50, num_epochs=10, batch_size=128):
        self.simulacrum_name = os.getenv("SIMULACRUM_NAME")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_words = max_words
        self.max_len = max_len
        self.tok = Tokenizer(num_words=max_words)
        self.processor = DataProcessor(os.getenv("SIMULACRUM_NAME"))
        self.model = self.architecture()
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    def architecture(self):
        inputs = Input(name='inputs', shape=[self.max_len])
        layer = Embedding(self.max_words, self.max_len, input_length=self.max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(self.max_len, name='out_layer')(layer)
        layer = Activation('relu')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    def architecture2(self):
        inputs = Input(name='inputs', batch_shape=(self.batch_size, self.max_len))
        layer = Embedding(self.max_words, self.max_len)(inputs)
        layer = GRU(1024, recurrent_initializer='glorot_uniform', stateful=True)(layer)
        layer = Dense(self.max_len, name='out_layer')(layer)
        # layer = Activation('relu')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model


    def tokenize_sentences(self, sentences):
        sequences = self.tok.texts_to_sequences(sentences)
        # sequences = []
        # for vector in self.tok.texts_to_sequences(sentences):
        #     sequences.append(np.interp(vector, (0, self.max_words), (0, 1)))
        return sequence.pad_sequences(sequences, maxlen=self.max_len)

    def detokenzie(self, vectors):
        return self.tok.sequences_to_texts((vectors*10000).astype("int"))
        # return self.tok.sequences_to_texts(np.interp(vectors, (0, 1), (0, self.max_words)).astype("int"))

    def create_inputs(self, sentences=None):
        if sentences is None:
            self.processor.extract()
            sentences = self.processor.received
        self.tok.fit_on_texts(sentences)
        # self.max_words = len(sentences)
        return self.tokenize_sentences(sentences)

    def generate(self, sentences=None):
        if sentences is None:
            inputs = self.create_inputs()
        else:
            inputs = self.create_inputs(sentences)
        return np.array(self.model.predict(inputs)), np.zeros(len(inputs))

    def train(self, callbacks=None):
        # cb = [EarlyStopping(monitor='val_loss', min_delta=0.0001)]
        cb=[]
        if callbacks is not None:
            cb.extend(callbacks)

        self.processor.extract()
        train_X = []
        train_y = []
        for pair in self.processor.pairs:
            train_X.append(self.processor.received[pair[1]])
            train_y.append(self.processor.sent[pair[0]])

        self.model.fit(self.create_inputs(train_X), self.create_inputs(train_y), epochs=self.num_epochs,
                       batch_size=self.batch_size, validation_split=0.2, callbacks=cb)


# generator = SimulacrumGenerator()
# outputs, y = generator.generate()
# print(outputs[0], generator.tokenize_sentences(generator.processor.received)[0])
# print(generator.detokenzie(outputs))