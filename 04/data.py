import os

import numpy
from numpy.fft import fft


from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import IterationScheme, SequentialScheme, ShuffledScheme
from fuel.transformers import Batch, Transformer
from scipy.ndimage.filters import gaussian_filter1d


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

test_set = H5PYDataset(HDF5_PATH, which_sets=('test',),
                                  sources=('example', 'eeg', 'acc'))
print "Test set:", test_set.num_examples


class Normalize(Transformer):
    def __init__(self, stream, todo, **kwargs):
        self.sources = stream.sources
        self.todo = {self.sources.index(i): j for i, j in todo.iteritems()}
        super(Normalize, self).__init__(stream, **kwargs)
    
    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')
        data = list(next(self.child_epoch_iterator))
        for i, axis in self.todo.iteritems():
            data[i] -= data[i].mean(axis=axis, keepdims=True)
            data[i] /= numpy.sqrt((data[i]**2).mean(axis=axis, keepdims=True))
        return tuple(data)

class FFT(Transformer):
    def __init__(self, stream, ffts, **kwargs):
        self.sources = stream.sources
        self.ffts = {self.sources.index(i): j for i, j in ffts.iteritems()}
        super(FFT, self).__init__(stream, **kwargs)
    
    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')
        data = list(next(self.child_epoch_iterator))
        for i, axis in self.ffts.iteritems():
            data[i] = numpy.sqrt(numpy.abs(fft(data[i], axis=axis)).astype('float32'))
            data[i] = numpy.nan_to_num(data[i])
        return tuple(data)

class GaussianFilter(Transformer):
    def __init__(self, stream, filters, **kwargs):
        self.sources = stream.sources
        self.filters = {self.sources.index(i): j for i, j in filters.iteritems()}
        super(GaussianFilter, self).__init__(stream, **kwargs)
    
    def get_data(self, request=None):
        if request is not None:
            raise ValueError('Unsupported: request')
        data = list(next(self.child_epoch_iterator))
        for i, (axis, sigma) in self.filters.iteritems():
            data[i] = gaussian_filter1d(data[i], axis=axis, sigma=sigma)
        return tuple(data)
            
def get_streams(batch_size):
    train_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(examples=train_set.num_examples, batch_size=batch_size))
    valid_stream = DataStream(valid_set, iteration_scheme=ShuffledScheme(examples=valid_set.num_examples, batch_size=batch_size))
    test_stream = DataStream(test_set, iteration_scheme=SequentialScheme(examples=test_set.num_examples, batch_size=batch_size))
    train_stream = Normalize(train_stream, {'eeg': 1, 'acc': 1})
    valid_stream = Normalize(valid_stream, {'eeg': 1, 'acc': 1})
    test_stream = Normalize(test_stream, {'eeg': 1, 'acc': 1})

    return train_stream, valid_stream, test_stream

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_stream = DataStream(train_set, iteration_scheme=ShuffledScheme(examples=train_set.num_examples, batch_size=1))
    train_stream = Normalize(train_stream, {'eeg': 1, 'acc': 1})

    for i, d in enumerate(train_stream.get_epoch_iterator(as_dict=True)):
        plt.plot(numpy.arange(d['acc'].shape[1]), d['acc'][0, :, 0], 'r')
        plt.plot(numpy.arange(d['acc'].shape[1]), d['acc'][0, :, 1], 'g')
        plt.plot(numpy.arange(d['acc'].shape[1]), d['acc'][0, :, 2], 'b')
        if i > 0: break
    plt.show()

    for i, d in enumerate(train_stream.get_epoch_iterator(as_dict=True)):
        color = ['r', 'g', 'b', 'k', 'm'][d['label']]
        if i in range(2,10):
            plt.plot(numpy.arange(d['eeg'].shape[1]), d['eeg'][0, :, 0], color)
        if i > 100:
            break
    plt.show()

    train_stream_fft = FFT(train_stream, {'eeg': 1, 'acc': 1})

    for i, d in enumerate(train_stream_fft.get_epoch_iterator(as_dict=True)):
        color = ['r', 'g', 'b', 'c', 'm'][d['label'][0]]
        print d['label']
        plt.plot(numpy.arange(d['acc'].shape[1]), d['acc'][0, :, 0], color)
        if i > 100:
            break
    plt.show()

    for i, d in enumerate(train_stream_fft.get_epoch_iterator(as_dict=True)):
        color = ['r', 'g', 'b', 'c', 'm'][d['label'][0]]
        print d['label']
        plt.plot(numpy.arange(d['eeg'].shape[1]), d['eeg'][0, :, 0], color)
        if i > 100:
            break
    plt.show()

# vim: set sts=4 ts=4 sw=4 tw=0 et :
