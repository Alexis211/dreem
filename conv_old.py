#!/usr/bin/env python

import theano
import numpy
from theano import tensor

from blocks.bricks import Linear, Tanh, Rectifier, Softmax, MLP, Identity
from blocks.bricks.conv import Convolutional, MaxPooling
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.algorithms import (GradientDescent, Scale, AdaDelta, RemoveNotFinite, RMSProp, BasicMomentum,
                               StepClipping, CompositeRule, Momentum)
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.main_loop import MainLoop

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS
from blocks.graph import ComputationGraph, apply_dropout, apply_noise

from blocks.extensions import ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing

from blocks.extras.extensions.plot import Plot


from data import get_streams

from ext_paramsaveload import SaveLoadParams


# ==========================================================================================
#                                     THE HYPERPARAMETERS
# ==========================================================================================

# Stop after this many epochs
n_epochs = 10000
# How often (number of batches) to print / plot
monitor_freq = 20

batch_size = 200


# regularization : noise on the weights
weight_noise = 0.01
dropout = 0.2

# number of classes, a constant of the dataset
num_output_classes = 5 


# the step rule (uncomment your favorite choice)
step_rule = CompositeRule([AdaDelta(), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.00001, momentum=0.99), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.1, momentum=0.9), RemoveNotFinite()])
#step_rule = CompositeRule([AdaDelta(), Scale(0.01), RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.1, decay_rate=0.95),
#                           RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.0001, decay_rate=0.95),
#                           BasicMomentum(momentum=0.9),
#                           RemoveNotFinite()])

# How the weights are initialized
weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.001)


# ==========================================================================================
#                                          THE MODEL
# ==========================================================================================

print('Building model ...')

def normalize(var, axis):
    var = var - var.mean(axis=axis, keepdims=True)
    var = var / tensor.sqrt((var**2).mean(axis=axis, keepdims=True))
    return var


#       THEANO INPUT VARIABLES
eeg = tensor.tensor3('eeg')
acc = tensor.tensor3('acc')
label = tensor.lvector('label')

# normalize
eeg = normalize(eeg, axis=1)

acc = normalize(acc, axis=1)

# set dims for convolution
eeg = eeg.dimshuffle(0, 2, 1, 'x')
acc = acc.dimshuffle(0, 2, 1, 'x')

# first convolution only on eeg
conv_eeg = Convolutional(filter_size=(300, 1),
                         num_filters=20,
                         num_channels=1,
                         border_mode='full',
                         tied_biases=True,
                         name="conv_eeg")
maxpool_eeg = MaxPooling(pooling_size=(5, 1), name='maxpool_eeg')
# convolve
eeg1 = conv_eeg.apply(eeg)
# cut borders
d1 = (eeg1.shape[2] - eeg.shape[2])/2
eeg1 = eeg1[:, :, d1:d1+eeg.shape[2], :]
# subsample
eeg1 = maxpool_eeg.apply(eeg1)
# activation
eeg1 = Tanh(name='act_eeg').apply(eeg1)

# second convolution only on eeg
conv_eeg2 = Convolutional(filter_size=(100, 1),
                         num_filters=40,
                         num_channels=20,
                         border_mode='full',
                         tied_biases=True,
                         name="conv_eeg2")
maxpool_eeg2 = MaxPooling(pooling_size=(5, 1), name='maxpool_eeg2')
# convolve
eeg2 = conv_eeg2.apply(eeg1)
# cut borders
d1 = (eeg2.shape[2] - eeg1.shape[2])/2
eeg2 = eeg2[:, :, d1:d1+eeg1.shape[2], :]
# subsample
eeg2 = maxpool_eeg.apply(eeg2)
# activation
eeg2 = Tanh(name='act_eeg2').apply(eeg2)

# Now we can concatenate eeg and acc (normally)
data = tensor.concatenate([eeg2, acc], axis=1)

# and do more convolutions
conv = Convolutional(filter_size=(50, 1),
                     num_filters=50,
                     num_channels=43,
                     border_mode='full',
                     tied_biases=True,
                     name="conv")
maxpool = MaxPooling(pooling_size=(3, 1), name='maxpool')
data1 = conv.apply(data)
# cut borders
d1 = (data1.shape[2] - data.shape[2])/2
data1 = data1[:, :, d1:d1+data.shape[2], :]
# max pool
data1 = maxpool.apply(data1)
# activation
data1 = Tanh(name='act_data').apply(data1)

# and do more convolutions
conv2 = Convolutional(filter_size=(30, 1),
                      num_filters=50,
                      num_channels=50,
                      border_mode='full',
                      tied_biases=True,
                      name="conv2")
maxpool2 = MaxPooling(pooling_size=(2, 1), name='maxpool2')
data2 = conv2.apply(data1)
# cut borders
d1 = (data2.shape[2] - data1.shape[2])/2
data2 = data2[:, :, d1:d1+data1.shape[2], :]
# max pool
data2 = maxpool2.apply(data2)
# activation
data2 = Tanh(name='act_data2').apply(data2)


# fully connected layers
fc = MLP(dims=[25*50, 100, 100, num_output_classes],
         activations=[Rectifier(name='r1'), Rectifier(name='r2'), Identity()])
output = fc.apply(data2.reshape((data2.shape[0], 25*50)))


#       COST AND ERROR MEASURE
cost = Softmax().categorical_cross_entropy(label, output).mean()
cost.name = 'cost'

error_rate = tensor.neq(tensor.argmax(output, axis=1), label).mean()
error_rate.name = 'error_rate'


#       REGULARIZATION
cg = ComputationGraph([cost, error_rate])
if weight_noise > 0:
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    cg = apply_noise(cg, noise_vars, weight_noise)
if dropout > 0:
    cg = apply_dropout(cg, [eeg1, eeg2, data1, data2] + VariableFilter(name='output', bricks=fc.linear_transformations[:-1])(cg), dropout)
# for vfilter, p in dropout_locs:
#     cg = apply_dropout(cg, vfilter(cg), p)
[cost_reg, error_rate_reg] = cg.outputs


#       INITIALIZATION
for brick in [conv_eeg, maxpool_eeg, conv_eeg2, maxpool_eeg2, conv, maxpool, conv2, maxpool2, fc]:
    brick.weights_init = weights_init
    brick.biases_init = biases_init
    brick.initialize()


# ==========================================================================================
#                                     THE INFRASTRUCTURE
# ==========================================================================================

#       SET UP THE DATASTREAM

print('Bulding DataStream ...')
stream, valid_stream = get_streams(batch_size)


#       SET UP THE BLOCKS ALGORITHM WITH EXTENSIONS

print('Bulding training process...')
algorithm = GradientDescent(cost=cost_reg,
                            parameters=ComputationGraph(cost).parameters,
                            step_rule=step_rule)

monitor_cost = TrainingDataMonitoring([cost_reg, error_rate_reg],
                                      prefix="train",
                                      every_n_batches=monitor_freq,
                                      after_epoch=False)

monitor_valid = DataStreamMonitoring([cost, error_rate],
                                     data_stream=valid_stream,
                                     prefix="valid",
                                     after_epoch=True)

plot = Plot(document='dreem_conv',
            channels=[['train_cost', 'valid_cost'],
                      ['train_error_rate', 'valid_error_rate']],
            every_n_batches=monitor_freq,
            after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[
                                 ProgressBar(),

                                 monitor_cost, monitor_valid,

                                 plot,
                                 Printing(every_n_batches=monitor_freq, after_epoch=True),

                                 SaveLoadParams('conv_params.pkl', Model(cost), before_training=True, after_epoch=True),

                                 FinishAfter(after_n_epochs=n_epochs),
                                ],
                     model=model)


#       NOW WE FINALLY CAN TRAIN OUR MODEL

print('Starting training ...')
main_loop.run()


# vim: set sts=4 ts=4 sw=4 tw=0 et:
