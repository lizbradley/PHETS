import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

from ROC.helpers import crop_sig, slice_sig, get_spec, downsample_spec, plot_sig

fname_1 = '../datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt'
fname_2 = '../datasets/time_series/viol/40-viol.txt'

label_1 = 'clarinet'
label_2 = 'viol'

crop_1 = (50000, 120000)
crop_2 = (100000, 170000)

window_length = 5000
num_windows = 50

FT_bins = 50




print 'loading...'
sig_full_1 = np.loadtxt(fname_1)
sig_full_2 = np.loadtxt(fname_2)

print 'processing...'
sig_1 = crop_sig(sig_full_1, crop_1)
sig_2 = crop_sig(sig_full_2, crop_2)


windows_1, st_pts = slice_sig(sig_1, l=window_length, n=num_windows)
windows_2, st_pts = slice_sig(sig_2, l=window_length, n=num_windows)

print 'plotting...'

plot_sig(sig_1, sig_full_1, windows_1, label_1)
plot_sig(sig_2, sig_full_2, windows_2, label_2)


specs_1 = [get_spec(w) for w in windows_1]
specs_2 = [get_spec(w) for w in windows_2]


plt.semilogx(*specs_1[0], basex=10)
plt.set_xlim = (10, 50000)
plt.savefig('spec.png')
plt.clf()

specs_ds_1 = np.asarray([downsample_spec(s[0], s[1], FT_bins) for s in specs_1])
specs_ds_2 = np.asarray([downsample_spec(s[0], s[1], FT_bins) for s in specs_2])

plt.semilogx(*specs_ds_1[0], basex=10)
plt.set_xlim =(10, 50000)
plt.savefig('spec_ds.png')
plt.clf()

specs = np.concatenate([specs_ds_1[:, 1], specs_ds_2[:, 1]])
labels = [label_1 for s in specs_ds_1] + [label_2 for s in specs_ds_2]


train_specs, test_specs, train_labels, test_labels = train_test_split(specs, labels, train_size=0.8)

clf = svm.SVC()
print 'training...'
clf.fit(train_specs, train_labels)

print 'testing...'
print clf.score(test_specs, test_labels)