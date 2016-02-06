import os
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset

def path(x):
    return os.path.join('/home/lx.nobackup/datasets/dreem', x)

print "Loading input_train..."
input_train = np.loadtxt(path('input_train.csv'), 
                         delimiter=',', skiprows=1, usecols=range(4, 4204))
output_train = np.loadtxt(path('output_train.csv'),
                          delimiter=';', skiprows=1, usecols=[1]).reshape((input_train.shape[0],))

print "Loading input_train again..."
input_train_ids = np.loadtxt(path('input_train.csv'),
                             delimiter=',', skiprows=1, usecols=[0], dtype='S15')

print "Loading input_test..."
input_test = np.loadtxt(path('input_test.csv'),
                        delimiter=',', skiprows=1, usecols=range(4, 4204))
print "Loading input_test again..."
input_test_ids = np.loadtxt(path('input_test.csv'),
                            delimiter=',', skiprows=1, usecols=[0], dtype='S15')

n_total = input_train.shape[0] + input_test.shape[0]

print "Writing HDF5 file"
f = h5py.File(path('dataset.hdf5'), mode='w')
example = f.create_dataset('example', (n_total, 1), dtype='S15')
eeg = f.create_dataset('eeg', (n_total, 3750, 1), dtype='float32')
acc = f.create_dataset('acc', (n_total, 150, 3), dtype='float32')
label = f.create_dataset('label', (output_train.shape[0],), dtype='int32')

example[:input_train.shape[0], 0] = input_train_ids
example[input_train.shape[0]:, 0] = input_test_ids
eeg[:input_train.shape[0], :, :] = input_train[:, :3750, None]
eeg[input_train.shape[0]:, :, :] = input_test[:, :3750, None]
acc[:input_train.shape[0], :, 0] = input_train[:, 3750:3900]
acc[:input_train.shape[0], :, 1] = input_train[:, 3900:4050]
acc[:input_train.shape[0], :, 2] = input_train[:, 4050:4200]
acc[input_train.shape[0]:, :, 0] = input_test[:, 3750:3900]
acc[input_train.shape[0]:, :, 1] = input_test[:, 3900:4050]
acc[input_train.shape[0]:, :, 2] = input_test[:, 4050:4200]
label[...] = output_train

eeg.dims[0].label = 'batch'
eeg.dims[1].label = 'time'
eeg.dims[2].label = 'feature'
acc.dims[0].label = 'batch'
acc.dims[1].label = 'time'
acc.dims[2].label = 'feature'
label.dims[0].label = 'batch'
label.dims[1].label = 'target'

b = input_train.shape[0]
c = n_total
split_dict = {
    'train': {'example': (0, b),
              'eeg':   (0, b),
              'acc':   (0, b),
              'label': (0, b)},
    'test': {'example': (b, c),
             'eeg':    (b, c),
             'acc':    (b, c)}}
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

print "flush close"
f.flush()
f.close()

# vim: set sts=4 ts=4 sw=4 tw=0 et :
