from keras.models import Sequential
from lstm_classifier import SimulacrumDiscriminator
from generator import SimulacrumGenerator
from keras.optimizers import Adam
import numpy as np
from processor import DataProcessor
import os
from dotenv import load_dotenv
load_dotenv()

class SolemnSimulacrum:
    def __init__(self, discriminator: SimulacrumDiscriminator, generator: SimulacrumGenerator):
        self.processor = discriminator.processor
        self.discriminator = discriminator
        self.generator = generator
        discriminator.trainable = False
        model = Sequential()
        model.add(generator.model)
        model.add(discriminator.model)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.model = model

    def train(self, num_epochs=100, batch_size=128):
        self.processor.extract()
        self.discriminator.tok.fit_on_texts(self.processor.sent)
        discriminator_X = self.discriminator.tokenize_sentences(self.processor.sent)
        discriminator_y = np.full(shape=len(self.processor.sent), fill_value=1, dtype=np.int)
        generator_X, generator_y = self.generator.generate(self.processor.received[0:len(discriminator_X)])
        gan_X, gan_y = self.generator.create_inputs(self.processor.received[0:len(discriminator_X)]), np.ones(len(discriminator_X))
        X, y = np.vstack((discriminator_X, generator_X)), np.hstack((discriminator_y, generator_y))

        for epoch in range(num_epochs):
            d_loss = self.discriminator.model.train_on_batch(X, y)[0]
            gan_loss = self.model.train_on_batch(gan_X, gan_y)
            print(f"Epoch {epoch} had training loss {gan_loss} for GAN and {d_loss} for Discriminator.")



# disc = SimulacrumDiscriminator()
# gen = SimulacrumGenerator()
# SolemnSimulacrum(disc, gen).train()