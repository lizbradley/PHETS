# from DCE.DCE import embed
import matplotlib.pyplot as plt
import numpy as np
# import scipy.fftpack as fftpack
#
#
# def get_spec(sig):
# 	sig_fft = fftpack.rfft(sig)
# 	spec = 20 * np.log10(np.abs(sig_fft))
# 	n = sig_fft.size
# 	timestep = 1 / 44100.
# 	freq = fftpack.rfftfreq(n, d=timestep)
#
# 	return [freq, spec]
#
#
#
# f = 50
# f = f * 2 * np.pi
# t = np.linspace(0, 1, 44100)
# # x = np.concatenate([np.sin(t * f), .5 * np.sin(t * f) + .5 * np.sin(t * f/2)])
# x = np.concatenate([np.sin(t * f), np.sin(t * 1.5 * f)])
# # x = np.sin(t * f)
# x = np.sin(t * f) + np.sin(t * 1.5 * f)
# plt.plot(x)
# plt.show()
#
#
# w, spec = get_spec(x)
#
# # # plt.semilogx(spec, lw=.5)
# # plt.plot(w, spec, lw=.5)
# # plt.xlim([10, 100])
# # plt.savefig('../sum.png')
#
# d = embed(x, tau=.001, m=2, time_units='seconds')
# plt.scatter(d[:,0], d[:,1], s=.1)
# plt.show()
#

def jittery_circle(n):
	import random
	jitter = np.array([random.uniform(0, .1) for i in range(n)])

	t = np.linspace(0, 2 * np.pi, n) + jitter
	x = np.sin(t)
	y = np.cos(t)
	return x, y

if __name__ == '__main__':
	# x, y = jittery_circle(10)

	t = np.array([0, 1./3, 1./2, 5./6, 1., 4./3, 3./2, 11./6]) * np.pi

	x = np.cos(t)
	y = np.sin(t)

	np.savetxt('datasets/trajectories/jittery_circle.txt', np.array([x, y]).T)

	plt.scatter(x, y)

	plt.axes().set_aspect('equal')
	plt.savefig('output/debug/jittery.png')
