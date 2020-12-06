from keras.models import Sequential, load_model
from lstm_classifier import SimulacrumDiscriminator
from generator import SimulacrumGenerator
from keras.optimizers import Adam
import numpy as np
from dotenv import load_dotenv
import os
import csv
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

        progress = ["epoch", "gan_loss", "discriminator_loss", "discriminator_accuracy", "sample_in", "sample_out"]
        with open("data/gan_progress.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(progress)

        for epoch in range(num_epochs):
            generator_X, generator_y = self.generator.generate(self.processor.received[0:len(discriminator_X)])
            gan_X, gan_y = self.generator.create_inputs(self.processor.received[0:len(discriminator_X)]), np.ones(len(discriminator_X))
            X, y = np.vstack((discriminator_X, generator_X)), np.hstack((discriminator_y, generator_y))

            d_loss, d_acc = self.discriminator.model.train_on_batch(X, y)
            gan_loss = self.model.train_on_batch(gan_X, gan_y)
            print(f"Epoch {epoch}: Training Loss - {gan_loss} for GAN. Training Loss - {d_loss}, Accuracy -  {d_acc} for Discriminator.")
            index = np.random.choice(len(self.processor.received), 1, replace=False)[0]
            sample_in = self.processor.received[index]
            sample_out = self.generator.generate([sample_in])[0]
            with open("data/gan_progress.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, gan_loss, d_loss, d_acc, sample_in, self.generator.detokenzie(sample_out)[0]])

    def save(self):
        self.model.save(f"data/SolemnSimulacrum_{self.generator.simulacrum_name}.model")
        self.discriminator.model.save(f"data/SimulacrumDiscriminator_{self.discriminator.simulacrum_name}.model")
        self.generator.model.save(f"data/SimulacrumGenerator_{self.generator.simulacrum_name}.model")

    def load(self, simulacrum_name=os.getenv("SIMULACRUM_NAME")):
        self.model = load_model(f"data/SolemnSimulacrum_{simulacrum_name}.model")
        self.discriminator.model = load_model(f"data/SimulacrumDiscriminator_{simulacrum_name}.model")
        self.generator.model = load_model(f"data/SimulacrumGenerator_{simulacrum_name}.model")



# disc = SimulacrumDiscriminator(max_len=15)
# gen = SimulacrumGenerator(max_len=15)
# simulacrum = SolemnSimulacrum(disc, gen)
# simulacrum.train(num_epochs=500)
# simulacrum.save()
