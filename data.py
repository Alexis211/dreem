import os

import numpy


from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, SequentialScheme, ShuffledScheme
from fuel.transformers import Batch


PATH = '/home/lx.nobackup/datasets/dreem'
HDF5_PATH = os.path.join(PATH, 'dataset.hdf5')

train_set = H5PYDataset(HDF5_PATH, which_sets=('train',),
						subset=slice(0,25000),
						sources=('acc', 'eeg', 'label')) 
print "Train set:", train_set.num_examples

valid_set = H5PYDataset(HDF5_PATH, which_sets=('train',),
						subset=slice(25000,31129),
						sources=('acc', 'eeg', 'label')) 
print "Valid set:", valid_set.num_examples

test_set = H5PYDataset(HDF5_PATH, which_sets=('test',))
print "Test set:", test_set.num_examples

def get_streams(batch_size):
	train_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(examples=train_set.num_examples, batch_size=batch_size))
	valid_stream = DataStream(valid_set, iteration_scheme=ShuffledScheme(examples=valid_set.num_examples, batch_size=batch_size))
	return train_stream, valid_stream


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	def normalize(x):
		x = x - x.mean(axis=0, keepdims=True)
		return x / numpy.sqrt((x**2).mean(axis=0, keepdims=True))
	def lognormalize(x):
		x = numpy.log(x)
		return normalize(x)

	train_stream = DataStream(train_set, iteration_scheme=SequentialScheme(examples=train_set.num_examples, batch_size=1))

	for i, d in enumerate(train_stream.get_epoch_iterator(as_dict=True)):
		plt.plot(numpy.arange(d['acc'].shape[1]), normalize(d['acc'][0, :, 0]), 'r')
		plt.plot(numpy.arange(d['acc'].shape[1]), normalize(d['acc'][0, :, 1]), 'g')
		plt.plot(numpy.arange(d['acc'].shape[1]), normalize(d['acc'][0, :, 2]), 'b')
		if i > 0: break
	plt.show()

	for i, d in enumerate(train_stream.get_epoch_iterator(as_dict=True)):
		plt.plot(numpy.arange(d['eeg'].shape[1]), normalize(d['eeg'][0, :, 0]))
		if i > 3: break
	plt.show()

