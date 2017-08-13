from DCE.DCE import embed
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftpack


def get_spec(sig):
	sig_fft = fftpack.rfft(sig)
	spec = 20 * np.log10(np.abs(sig_fft))
	n = sig_fft.size
	timestep = 1 / 44100.
	freq = fftpack.rfftfreq(n, d=timestep)

	return [freq, spec]



f = 50
f = f * 2 * np.pi
t = np.linspace(0, 1, 44100)
# x = np.concatenate([np.sin(t * f), .5 * np.sin(t * f) + .5 * np.sin(t * f/2)])
x = np.concatenate([np.sin(t * f), np.sin(t * 1.5 * f)])
# x = np.sin(t * f)
x = np.sin(t * f) + np.sin(t * 1.5 * f)
plt.plot(x)
plt.show()


w, spec = get_spec(x)

# # plt.semilogx(spec, lw=.5)
# plt.plot(w, spec, lw=.5)
# plt.xlim([10, 100])
# plt.savefig('../sum.png')

d = embed(x, tau=.001, m=2, time_units='seconds')
plt.scatter(d[:,0], d[:,1], s=.1)
plt.show()
