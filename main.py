from lstm_classifier import SimulacrumDiscriminator
from generator import SimulacrumGenerator
from processor import DataProcessor
from simulacrum_gan import SolemnSimulacrum
from keras.utils.vis_utils import plot_model

plot_model(SimulacrumDiscriminator().model, to_file='data/discriminator_plot.png', show_shapes=True, show_layer_names=True)
plot_model(SimulacrumGenerator().model, to_file='data/generator_plot.png', show_shapes=True, show_layer_names=True)
