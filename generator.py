import numpy as np
from keras.models import Model, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from processor import DataProcessor
import os
from dotenv import load_dotenv
load_dotenv()


class SimulacrumGenerator:

    def __init__(self, max_words=1000, max_len=150, num_epochs=10, batch_size=128):
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
        layer = Embedding(self.max_words, 50, input_length=self.max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation('tanh')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(self.max_len, name='out_layer')(layer)
        # layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model


    def tokenize_sentences(self, sentences):
        sequences = self.tok.texts_to_sequences(sentences)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)

    def detokenzie(self, vectors):
        return self.tok.sequences_to_texts(vectors)

    def create_inputs(self, sentences=None):
        if sentences is None:
            self.processor.extract()
            sentences = self.processor.received
        self.tok.fit_on_texts(sentences)
        return self.tokenize_sentences(sentences)

    def generate(self, sentences=None):
        if sentences is None:
            inputs = self.create_inputs()
        else:
            inputs = self.create_inputs(sentences)
        return np.array(self.model.predict(inputs)), np.zeros(len(inputs))

# generator = SimulacrumGenerator()
# outputs, y = generator.generate()
# print(outputs[0], generator.tokenize_sentences(generator.processor.received)[0])
# print(generator.detokenzie(outputs))