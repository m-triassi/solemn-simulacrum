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


class SimulacrumDiscriminator:

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
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model

    def tokenize_sentences(self, sentences):
        sequences = self.tok.texts_to_sequences(sentences)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)

    def train(self, train_X=None, train_y=None):
        if train_X is None and train_y is None:
            self.train_X, self.test_X, self.train_y, self.test_y = self.processor.plain_label()
            train_X, train_y = self.train_X, self.train_y
        self.tok.fit_on_texts(train_X)
        self.fit(self.tokenize_sentences(train_X), train_y)
        return self

    def fit(self, sequences_matrix, train_y):
        self.model.fit(sequences_matrix, train_y, batch_size=self.batch_size, epochs=self.num_epochs,
                  validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        return self

    def evaluate(self, test_X=None, test_y=None):
        if (test_X is None and test_y is None):
            self.train_X, self.test_X, self.train_y, self.test_y = self.processor.plain_label()
            test_X, test_y = self.test_X, self.test_y
        accr = self.model.evaluate(self.tokenize_sentences(test_X), test_y)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

    def predict(self, sentences):
        return self.model.predict(self.tok.texts_to_sequences(sentences))

    def save(self, train_X=None, test_X=None, train_y=None, test_y=None, with_data=False):
        self.model.save(f"data/SimulacrumDiscriminator_{self.simulacrum_name}.model")
        settings = np.array([self.num_epochs, self.batch_size, self.max_words, self.max_len])
        print(settings)
        np.savetxt(f"data/SD_settings_{self.simulacrum_name}.csv", settings, delimiter=",")
        if with_data:
            if train_X is None and test_X is None and train_y is None and test_y is None:
                self.processor.cache_results(self.train_X, self.test_X, self.train_y, self.test_y)
            else:
                self.processor.cache_results(train_X, test_X, train_y, test_y)

    def load(self, simulacrum_name=os.getenv("SIMULACRUM_NAME"), with_data=False):
        if simulacrum_name is None: simulacrum_name = os.getenv("SIMULACRUM_NAME")
        self.model = load_model(f"data/SimulacrumDiscriminator_{simulacrum_name}.model")
        self.num_epochs, self.batch_size, self.max_words, self.max_len = np.loadtxt(f"data/SD_settings_{self.simulacrum_name}.csv", delimiter=",")
        self.simulacrum_name = simulacrum_name
        if with_data:
            self.train_X, self.test_X, self.train_y, self.test_y = self.processor.load_cache(simulacrum_name)
        return self




# discriminator = SimulacrumDiscriminator(batch_size=256).train()
# discriminator.save()
# discriminator = SimulacrumDiscriminator().load()

