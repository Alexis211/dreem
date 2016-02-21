#!/usr/bin/env python

import theano
import numpy
import sys

import csv

from theano import tensor
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from blocks.bricks import Linear, Tanh, Rectifier, Softmax, MLP, Identity
from blocks.bricks.conv import Convolutional, MaxPooling
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import IsotropicGaussian, Constant

from blocks.algorithms import (GradientDescent, Scale, AdaDelta, RemoveNotFinite, RMSProp, BasicMomentum, Adam,
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


from data import get_streams, FFT, GaussianFilter

from ext_paramsaveload import SaveLoadParams


# ==========================================================================================
#                                     THE HYPERPARAMETERS
# ==========================================================================================

# Stop after this many epochs
n_epochs = 10000
# How often (number of batches) to print / plot
monitor_freq = 20

batch_size = 200

data_dropout = 0

out_hidden = [100, 100, 50]
out_activation = [Rectifier, Rectifier, Rectifier]
out_dropout = 0.8

# regularization : noise on the weights
weight_noise = 0. # 0.0001
# regularization : L1
l1_reg = 0.0001
l2_reg = 0.0000

# number of classes, a constant of the dataset
num_output_classes = 5 


# the step rule (uncomment your favorite choice)
step_rule = CompositeRule([AdaDelta(), RemoveNotFinite()])
#step_rule = AdaDelta()
#step_rule = CompositeRule([Momentum(learning_rate=0.00001, momentum=0.99), RemoveNotFinite()])
#step_rule = CompositeRule([Momentum(learning_rate=0.01, momentum=0.9), RemoveNotFinite()])
#step_rule = CompositeRule([AdaDelta(), Scale(0.01), RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.1, decay_rate=0.95),
#                           RemoveNotFinite()])
#step_rule = CompositeRule([RMSProp(learning_rate=0.001, decay_rate=0.95),
#                           BasicMomentum(momentum=0.9),
#                           RemoveNotFinite()])
#step_rule = Adam()

# How the weights are initialized
weights_init = IsotropicGaussian(0.01)
biases_init = Constant(0.001)


# ==========================================================================================
#                                          THE MODEL
# ==========================================================================================

print('Building model ...')

bricks = []
dropout_locs = []

#       THEANO INPUT VARIABLES
eeg = tensor.tensor3('eeg')         # batch x time x feature
acc = tensor.tensor3('acc')         # batch x time x feature
label = tensor.lvector('label')     # batch

eeg_len = 150*25
acc_len = 150
acc_chan = 3

def normalize(var, axis):
    var = var - var.mean(axis=axis, keepdims=True)
    var = var / tensor.sqrt((var**2).mean(axis=axis, keepdims=True))
    return var
eeg1 = normalize(eeg, axis=0)
eeg2 = normalize(eeg, axis=1)
eeg3 = normalize(eeg1, axis=1)
eeg4 = normalize(eeg2, axis=0)

acc1 = normalize(acc, axis=0)
acc2 = normalize(acc, axis=1)
acc3 = normalize(acc1, axis=1)
acc4 = normalize(acc2, axis=0)
[eeg1, eeg2, eeg3, eeg4] = [x.reshape((-1, eeg_len)) for x in [eeg1, eeg2, eeg3, eeg4]]
[acc1, acc2, acc3, acc4] = [x.reshape((-1, acc_len*acc_chan)) for x in [acc1, acc2, acc3, acc4]]

eegs = [eeg1, eeg3]
accs = [acc1, acc3]


# fully connected layers
data = tensor.concatenate(eegs + accs, axis=1)
if data_dropout > 0:
    dropout_locs += [(lambda x: [data], data_dropout)]
data_sz = len(eegs) * eeg_len + len(accs) * acc_len * acc_chan

fc = MLP(dims=[data_sz] + out_hidden + [num_output_classes],
         activations=[r(name='out_act%d'%i) for i, r in enumerate(out_activation)] + [Identity()])
output = fc.apply(data)
if out_dropout > 0:
    dropout_locs += [(VariableFilter(name='output', bricks=fc.linear_transformations[:-1]), out_dropout)]


#       COST AND ERROR MEASURE
cost = Softmax().categorical_cross_entropy(label, output).mean()
cost.name = 'cost'

pred = tensor.argmax(output, axis=1)
pred.name = 'pred'

error_rate = tensor.neq(pred, label).mean()
error_rate.name = 'error_rate'


#       REGULARIZATION
cg = ComputationGraph([cost, error_rate])
if weight_noise > 0:
    noise_vars = VariableFilter(roles=[WEIGHT])(cg)
    cg = apply_noise(cg, noise_vars, weight_noise)
for loc, p in dropout_locs:
    ct = apply_dropout(cg, loc(cg), p)
[cost_reg, error_rate_reg] = cg.outputs
if l1_reg > 0:
    cost_reg += l1_reg * sum(x.norm(1) for x in VariableFilter(roles=[WEIGHT])(cg))
if l2_reg > 0:
    cost_reg += l2_reg * sum(x.norm(2) for x in VariableFilter(roles=[WEIGHT])(cg))
cost_reg.name = 'cost'


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
stream, valid_stream, test_stream = get_streams(batch_size)
stream, valid_stream, test_stream = [
            FFT(x, {'eeg': 1, 'acc': 1})
            for x in [stream, valid_stream, test_stream]]
stream, valid_stream, test_stream = [
            GaussianFilter(x, {'eeg': (1,10), 'acc': (1,5)})
            for x in [stream, valid_stream, test_stream]]


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

plot = Plot(document='dreem_fftmlp',
            channels=[['train_cost', 'valid_cost'],
                      ['train_error_rate', 'valid_error_rate']],
            every_n_batches=monitor_freq,
            after_epoch=True)

saveload = SaveLoadParams('fftmlp_params.pkl', Model(cost), before_training=True, after_epoch=True)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[
                                 ProgressBar(),

                                 monitor_cost, monitor_valid,

                                 plot,
                                 Printing(every_n_batches=monitor_freq, after_epoch=True),

                                 #saveload,

                                 FinishAfter(after_n_epochs=n_epochs),
                                ],
                     model=model)


#       NOW WE FINALLY CAN TRAIN OUR MODEL

if not '--notrain' in sys.argv:
    print('Starting training ...')
    main_loop.run()
else:
    saveload.do_load()

if '--eval' in sys.argv:
    print('Evaluating model on test set')

    # and evaluate it on the test set
    f = ComputationGraph(pred).get_theano_function()
    with open('test_output.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['ID','TARGET'])
        for i, x in enumerate(test_stream.get_epoch_iterator(as_dict=True)):
            print "Batch %d..."%i
            [pred] = f(acc=x['acc'], eeg=x['eeg'])
            for i in range(len(x['example'])):
                csvwriter.writerow([x['example'][i,0], "%.1f"%pred[i]])


# vim: set sts=4 ts=4 sw=4 tw=0 et:
