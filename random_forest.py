import gc
from datetime import datetime

import h5py
import numpy
import random
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import ricker, fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Results :
# - fft eeg, random forest : 33% error
# - fft eeg+acc, random forest : 32% error
# - normalize time, fft eeg + acc, random forest : 34% error


f = h5py.File('/home/lx.nobackup/datasets/dreem/dataset.hdf5')

def center(var, axis):
    return var - var.mean(axis=axis, keepdims=True)

def normalize(var, axis):
    var = center(var, axis)
    var /= numpy.sqrt((var**2).mean(axis=axis, keepdims=True)+1e-10)
    return var


def my_wavelet(a, b):
    r = numpy.arange(-a/2, a/2, dtype='float32')
    return numpy.exp(2j*r/b - (r**2)/(b**2)/2)

use_pca = True
classwise_pca = True

#eegwavelets = [(a, my_wavelet(3750, a*1.25)) for a in (2.**(1./2))**(numpy.arange(0,20))]      # NO
#eegwavelets = [(a, ricker(3750, a)) for a in (2.)**(numpy.arange(0, 12))]
eegwavelets = [(a, ricker(3750, a)) for a in (2.**(1./2))**(numpy.arange(0, 22))]    # best config?
#eegwavelets = [(a, ricker(3750, a*1.25/2.)) for a in (2.**(1./2))**(numpy.arange(0, 20))]
#eegwavelets = [(a, ricker(3750, a*1.25/2.)) for a in [1, 4, 16, 64, 256, 1024]] # test
#eegwavelets = [(a, ricker(3750, a*1.25/2.)) for a in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]]
#eegwavelets = [(a, ricker(3750, a)) for a in (2.**(1./4))**(numpy.arange(0, 42))]
eegpca = []

wfft = fft(numpy.concatenate([b[None, :] for a, b in eegwavelets], axis=0))
for i in range(len(eegwavelets)):
    plt.plot(numpy.arange(3750), eegwavelets[i][1])
plt.savefig('__wavelets.png')
plt.clf()
for i in range(wfft.shape[0]):
    plt.plot(numpy.arange(3750), numpy.abs(wfft[i]))
plt.savefig('__wavelets_fft.png')
plt.clf()

#accwavelets = [(a, my_wavelet(150, a*1.25)) for a in (2.)**(numpy.arange(0, 16))]      # NO
#accwavelets = [(a, my_wavelet(150, a*1.25)) for a in (2.)**(numpy.arange(0, 8))]
#accwavelets = [(a, ricker(150, a)) for a in (2.)**(numpy.arange(0, 8))]
#accwavelets = [(a, ricker(150, a*1.25)) for a in (2.)**(numpy.arange(0, 16))]            # best config?
#accwavelets = [(a, ricker(150, a)) for a in (2.**(1./4))**(numpy.arange(0, 28))]
#accwavelets = [(a, ricker(150, a*1.25/2.)) for a in (2.**(1./2))**(numpy.arange(0, 14))]
accwavelets = [(a*1.25, ricker(150, a*1.25)) for a in (2.)**(numpy.arange(0, 12))]
accpca = []

def process(eeg, acc, fit_pca=False):
    n = eeg.shape[0]

    print "Preprocess", n, "rows:"
    def subprocess(sig, wavelets, pcas, gauss_sigma, subsample, components):
        print "- FFT"
        sigfft = fft(sig, axis=1)
        ret = []
        for i, (a, wv) in enumerate(wavelets):
            gc.collect(2)
            print "- Wavelet Transform", i+1,"/",len(wavelets)
            wvfft = fft(wv[None, :], axis=1)
            if len(sigfft.shape) == 3: wvfft = wvfft[:, :, None]
            sig1 = sigfft * wvfft
            print "  - abs(ifft)"
            sig1 = numpy.abs(ifft(sig1, axis=1))
            print "  - fft"
            sig1 = numpy.sqrt(numpy.abs(fft(sig1, axis=1)))
            print "  - Filter..."
            sig1 = gaussian_filter1d(sig1, axis=1, sigma=gauss_sigma)
            sig1 = sig1.reshape((n, -1))
            if use_pca:
                if fit_pca:
                    print   "  - Fit PCA"
                    pcas.append(PCA(n_components=components, copy=True))
                    pcas[i].fit(sig1[:, ::subsample])
                print "  - PCA"
                sig1 = pcas[i].transform(sig1[:, ::subsample])
                ret.append(sig1)
            else:
                ret.append(numpy.array(sig1[:, ::subsample]))
        return ret

    eegs = subprocess(eeg, eegwavelets, eegpca, 30, 10, 15)
    accs = [x*0.05 for x in subprocess(acc, accwavelets, accpca, 10, 5, 10)]

    return numpy.concatenate(eegs+accs, axis=1)

n = 25000
p = 25000+6129
k = 25000+6129

wtrain_x = process(f['eeg'][:p, :, 0], f['acc'][:p, :, :], True)
wtrain_y = numpy.array(f['label'])

print "Visualize"
for i in range(10, 120, 5):
    color = ['r', 'g', 'b', 'k', 'm'][wtrain_y[i]]
    plt.plot(numpy.arange(wtrain_x.shape[1]), wtrain_x[i, :], color)
plt.savefig('__features.png')
plt.clf()

print "Shuffling train set..."
order = list(range(p))
random.shuffle(order)
wtrain_x = numpy.concatenate([wtrain_x[i, :][None, :] for i in order], axis=0)
wtrain_y = numpy.array([wtrain_y[i] for i in order])

print "Cross-validate..."
splits = range(0, p, p/3+1) + [p]
rfs = []
for i in range(1, len(splits)):
    a, b = splits[i-1:i+1]
    print "- split", a, "-", b

    valid_x = wtrain_x[a:b,:]
    valid_y = wtrain_y[a:b]

    train_x = numpy.concatenate([wtrain_x[0:a,:],wtrain_x[b:,:]], axis=0)
    train_y = numpy.concatenate([wtrain_y[0:a],wtrain_y[b:]], axis=0)

    rf = RandomForestClassifier(
            n_estimators=128,
            min_samples_split=5,
            n_jobs=4,
            max_features='log2',
            )
    rf.fit(train_x, train_y)
    rfs += [rf]

    tre = numpy.not_equal(rf.predict(train_x), train_y).mean()
    print "  train error rate:",tre
    valid_yhat = rf.predict(valid_x)

    p0 = numpy.equal(valid_yhat, valid_y).mean()
    vae = 1-p0
    pe = 0
    for i in range(5):
        a1 = numpy.equal(valid_y, i).mean()
        a2 = numpy.equal(valid_yhat, i).mean()
        pe += a1 * a2
    kappa = (p0-pe)/(1-pe)
    score=(1+kappa)/2
    print "  valid error rate:", vae, ", kappa:", kappa, ", score:", score

print "Predicting for test:"
test_ids = f['example'][k:]
test_x = process(f['eeg'][k:, :, 0], f['acc'][k:, :, :])
test_yhat = sum(numpy.equal(rf.predict(test_x)[:,None], numpy.arange(5)[None, :]) for rf in rfs).argmax(axis=1)

with open('rf_test_output-%s-%.4f-%.4f.csv'%(datetime.now().strftime('%s'), vae, score), 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['ID','TARGET'])
    for i in range(len(test_yhat)):
        csvwriter.writerow([test_ids[i,0], "%.1f"%test_yhat[i]])

# vim: set sts=4 ts=4 sw=4 tw=0 et :
