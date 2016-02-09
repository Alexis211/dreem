import os

import numpy


from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, SequentialScheme


PATH = '/home/lx.nobackup/datasets/dreem'
HDF5_PATH = os.path.join(PATH, 'dataset.hdf5')

train_set = H5PYDataset(HDF5_PATH, which_sets=('train',), subset=slice(0,25000)) 
print "Train set:", train_set.num_examples

valid_set = H5PYDataset(HDF5_PATH, which_sets=('train',), subset=slice(25000,31130)) 
print "Valid set:", valid_set.num_examples

test_set = H5PYDataset(HDF5_PATH, which_sets=('test',))
print "Test set:", test_set.num_examples


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	train_stream = DataStream(train_set, iteration_scheme=SequentialScheme(examples=train_set.num_examples, batch_size=1))
	for i, d in enumerate(train_stream.get_epoch_iterator(as_dict=True)):
		plt.plot(numpy.arange(d['acc'].shape[1]), numpy.log(d['acc'][0, :, 0]), 'r')
		plt.plot(numpy.arange(d['acc'].shape[1]), numpy.log(d['acc'][0, :, 1]), 'g')
		plt.plot(numpy.arange(d['acc'].shape[1]), numpy.log(d['acc'][0, :, 2]), 'b')
		if i > 0: break

	plt.show()

