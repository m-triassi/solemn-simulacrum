from lstm_classifier import SimulacrumDiscriminator
from generator import SimulacrumGenerator
from processor import DataProcessor
from simulacrum_gan import SolemnSimulacrum
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger
import random
import json

discriminator = SimulacrumDiscriminator(max_len=15)
generator = SimulacrumGenerator(max_len=15)
simulacrum = SolemnSimulacrum(discriminator, generator)

# d_logger = CSVLogger("data/discriminator.log")
# discriminator.train(callbacks=[d_logger])

# g_logger = CSVLogger("data/generator.log")
# generator.train(callbacks=[g_logger])
# generator.processor.extract()
# random_index = random.choice(range(0, len(generator.processor.received)-10))
# received_slice = generator.processor.received[random_index:random_index+10]
# received_slice.extend(generator.detokenzie(generator.generate(received_slice)[0]))
# with open('data/generated_responses.txt', 'w') as filehandle:
#     json.dump(received_slice, filehandle)
#
# plot_model(discriminator.model, to_file='data/discriminator_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(generator.model, to_file='data/generator_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(SolemnSimulacrum(discriminator, generator).model, to_file='data/gan_plot.png', show_shapes=True, show_layer_names=True)


simulacrum.load()
results = simulacrum.generator.generate(["anyway, i gotta eat now, it was nice talking to you :)"])[0]
print(results)
print(simulacrum.generator.detokenzie(results))