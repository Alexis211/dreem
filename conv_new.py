#!/usr/bin/env python

import theano
import numpy

from theano import tensor
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

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

eeg_gaussian_filter_width = 0 # 21
eeg_gaussian_filter_sigma = 4
eeg_gaussian_filter_step = 1

conv_eeg = [
    {'filter_size': 301,
     'num_filters': 50,
     'pool_size': 5,
     'activation': Tanh,
     'normalize': False,
    },
    {'filter_size': 21,
     'num_filters': 80,
     'pool_size': 5,
     'activation': Rectifier,
     'normalize': True,
    },
]

conv_all = [
    {'filter_size': 51,
     'num_filters': 100,
     'pool_size': 3,
     'activation': Tanh,
     'normalize': False,
    },
    {'filter_size': 15,
     'num_filters': 100,
     'pool_size': 2,
     'activation': Rectifier,
     'normalize': True,
    },
]

out_hidden = [500]
out_activation = [Rectifier]


# regularization : noise on the weights
weight_noise = 0.01
dropout = 0.5

# number of classes, a constant of the dataset
num_output_classes = 5 


# the step rule (uncomment your favorite choice)
#step_rule = CompositeRule([AdaDelta(), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.00001, momentum=0.99), RemoveNotFinite()])
step_rule = CompositeRule([Momentum(learning_rate=0.1, momentum=0.9), RemoveNotFinite()])
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


bricks = []
dropout_locs = []

#       THEANO INPUT VARIABLES
eeg = tensor.tensor3('eeg')         # batch x time x feature
acc = tensor.tensor3('acc')         # batch x time x feature
label = tensor.lvector('label')     # batch

# normalize
eeg = normalize(eeg, axis=1)
acc = normalize(acc, axis=1)

# set dims for convolution
eeg = eeg.dimshuffle(0, 2, 1, 'x')
acc = acc.dimshuffle(0, 2, 1, 'x')

# apply gaussian filter
if eeg_gaussian_filter_width > 0:
    l = eeg_gaussian_filter_width/2
    kernel = numpy.exp(-(numpy.arange(-l, l)**2)/(2*eeg_gaussian_filter_sigma**2))
    kernel = kernel / numpy.sqrt(2*3.1415) / eeg_gaussian_filter_sigma
    kernel = kernel.astype('float32')
    eeg1 = conv2d(eeg[:, 0, :, :], kernel[:, None], border_mode='full')[:, None, :, :]
    d1 = (eeg1.shape[2] - eeg.shape[2])/2
    eeg = eeg1[:, :, d1:d1+eeg.shape[2]:eeg_gaussian_filter_step, :]

# first convolutions only on eeg
eeg_channels = 1
for i, cp in enumerate(conv_eeg):
    bconv = Convolutional(filter_size=(cp['filter_size'], 1),
                          num_filters=cp['num_filters'],
                          num_channels=eeg_channels,
                          border_mode='full',
                          tied_biases=True,
                          name="conv_eeg_%d"%i)
    bmaxpool = MaxPooling(pooling_size=(cp['pool_size'], 1), name='maxpool_eeg_%d'%i)
    # convolve
    eeg1 = bconv.apply(eeg)
    # cut borders
    d1 = (eeg1.shape[2] - eeg.shape[2])/2
    eeg1 = eeg1[:, :, d1:d1+eeg.shape[2], :]
    # subsample
    eeg1 = bmaxpool.apply(eeg1)
    # normalize
    if cp['normalize']:
        eeg1 = normalize(eeg1, axis=(0, 2))
    # activation
    eeg1 = cp['activation'](name='act_eeg%d'%i).apply(eeg1)
    # stuff
    bricks += [bconv, bmaxpool]
    eeg = eeg1
    eeg_channels = cp['num_filters']
    dropout_locs += [eeg]


# Now we can concatenate eeg and acc (normally)
data = tensor.concatenate([eeg, acc], axis=1)
data_channels = eeg_channels + 3
data_len = 150

# and do more convolutions
for i, cp in enumerate(conv_all):
    conv = Convolutional(filter_size=(cp['filter_size'], 1),
                         num_filters=cp['num_filters'],
                         num_channels=data_channels,
                         border_mode='full',
                         tied_biases=True,
                         name="conv%d"%i)
    maxpool = MaxPooling(pooling_size=(cp['pool_size'], 1), name='maxpool%d'%i)
    data1 = conv.apply(data)
    # cut borders
    d1 = (data1.shape[2] - data.shape[2])/2
    data1 = data1[:, :, d1:d1+data.shape[2], :]
    # max pool
    data1 = maxpool.apply(data1)
    # normalize
    if cp['normalize']:
        data1 = normalize(data1, axis=(0, 2))
    # activation
    data1 = cp['activation'](name='act_data%d'%i).apply(data1)
    # stuff
    bricks += [conv, maxpool]
    data = data1
    data_channels = cp['num_filters']
    data_len /= cp['pool_size']
    dropout_locs += [data]


# fully connected layers
fc = MLP(dims=[data_len*data_channels] + out_hidden + [num_output_classes],
         activations=[r(name='fact%d'%i) for i, r in enumerate(out_activation)] + [Identity()])
output = fc.apply(data.reshape((data.shape[0], data_len*data_channels)))


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
    cg = apply_dropout(cg, dropout_locs + VariableFilter(name='output', bricks=fc.linear_transformations[:-1])(cg), dropout)
# for vfilter, p in dropout_locs:
#     cg = apply_dropout(cg, vfilter(cg), p)
[cost_reg, error_rate_reg] = cg.outputs


#       INITIALIZATION
for brick in bricks + [fc]:
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

plot = Plot(document='dreem_conv F%d,%d,%d ConvEEG%s%s%s Conv%s%s%s Out%s Noise%f Dropout%f' %
                    (eeg_gaussian_filter_width,eeg_gaussian_filter_sigma,eeg_gaussian_filter_step,
                     repr([x['filter_size'] for x in conv_eeg]),
                     repr([x['num_filters'] for x in conv_eeg]),
                     repr([x['pool_size'] for x in conv_eeg]),
                     repr([x['filter_size'] for x in conv_all]),
                     repr([x['num_filters'] for x in conv_all]),
                     repr([x['pool_size'] for x in conv_all]),
                     repr(out_hidden),
                     weight_noise,
                     dropout,
                    ),
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

                                 #SaveLoadParams('conv_params.pkl', Model(cost), before_training=True, after_epoch=True),

                                 FinishAfter(after_n_epochs=n_epochs),
                                ],
                     model=model)


#       NOW WE FINALLY CAN TRAIN OUR MODEL

print('Starting training ...')
main_loop.run()


# vim: set sts=4 ts=4 sw=4 tw=0 et:
