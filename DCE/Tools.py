import sys
import os
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, lfilter, freqz

WAV_SAMPLE_RATE = 44100.


def auto_crop(crop_cmd, sig, length):
	""" 
		finds max of volume envelope: (xmax, ymax)
		gets first point (x, y) on envelope where y < .1 * ymax and x > xmax 
	"""

	# lowpass from http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
	def butter_lowpass(cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False)
		return b, a

	def butter_lowpass_filter(data, cutoff, fs, order=5):
		b, a = butter_lowpass(cutoff, fs, order=order)
		y = lfilter(b, a, data)
		return y

	def show():
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(t, sig, label='data', color='lightblue')
		ax.plot(t, envelope, linewidth=1, label='filtered data', color='red')
		ax.set_xlabel('Time [sec]')
		ax.grid()
		ax.legend()

		ax.plot(t[max_arg], envelope[max_arg], color='k', marker='.')
		print t[max_arg], envelope[max_arg]

		ax.plot(t[st_arg], envelope[st_arg], color='k', marker='.')

		plt.show()


	# check input #
	if isinstance(crop_cmd, basestring):
		if not crop_cmd == 'auto':
			print "ERROR: embed crop option not recognized. Use 'auto' or set explicitly, e.g.:"
			print "\tembed_crop_1 = 'auto'"
			print "\tembed_crop_2 = (1, 1.3)"
			sys.exit()


	if crop_cmd == 'auto':
		sig_abs = np.abs(sig)
		order, fs, cutoff = 1, WAV_SAMPLE_RATE, 1  # filter params
		envelope = butter_lowpass_filter(sig_abs, cutoff, fs, order)

		n = len(sig)
		T = n / fs
		t = np.linspace(0, T, n, endpoint=False)

		max_arg = np.argmax(envelope)
		max = envelope[max_arg]

		st_arg = 0
		for i in xrange(max_arg, len(envelope)):
			if envelope[i] < .1 * max:
				st_arg = i
				break

		st_t = st_arg/WAV_SAMPLE_RATE
		crop = (st_t, st_t + length)
		print 'auto crop: ({:.3f}, {:.3f})'.format(crop[0], crop[1])


	else:
		crop = crop_cmd

	return crop


def auto_tau(tau_cmd, sig, note_index, tau_T, crop, filename):
	""" 
		helper for PRFCompare.mean_PRF_dist_plots() and DCE.compare_multi()
	"""

	# check input #
	if isinstance(tau_cmd, basestring):

		if not note_index:
			print "ERROR: 'note_index' required for tau='auto detect'"

		if tau_cmd not in ('auto detect', 'auto ideal'):
			print 'ERROR: tau_cmd not recognized.'
			sys.exit()

		fname_index = int(filename.split('/')[-1].split('-')[0])
		if fname_index != note_index:
			r = input(
				"WARNING: 'note_index' does not seem to match fname_index. 'note_index' is required for auto tau functionality. Continue? (y/n)")
			if r == 'n':
				sys.exit()


	# do it #
	ideal_freq = math.pow(2, (40 - float(note_index)) / 12) * 440  # Hz, descending index

	if not isinstance(tau_cmd, basestring):
		tau = tau_cmd
		f_disp = 'none'

	elif tau_cmd == 'auto detect':
		f = get_fund_freq(sig, ideal_freq, crop)
		T = 1. / f
		tau_sec = tau_T * T
		tau = tau_sec
		f_disp = f

	elif tau_cmd == 'auto ideal':
		f = ideal_freq
		T = 1. / f
		tau_sec = tau_T * T
		tau = tau_sec
		f_disp = 'none'


	print 'auto tau: {:.5f} (sec), {:d} (samples)'.format(tau, int(tau * WAV_SAMPLE_RATE))
	return ideal_freq, f_disp, tau



def plot_power_spectrum(sig, out_file, crop=(1,2)):
	from Plotter import plot_waveform
	samp_freq = 44100.

	if crop != 'none':
		window = np.array(crop) * samp_freq
		sig_crop = sig[int(window[0]):int(window[1])]
	else:
		sig_crop = sig
	FFT = scipy.fftpack.fft(sig_crop)
	FFT = 20 * scipy.log10(scipy.absolute(FFT)) # convert to db
	FFT_x = scipy.fftpack.fftfreq(len(FFT), d=1 / samp_freq)

	fig, subplots = plt.subplots(2, figsize=(6, 3), dpi=300, tight_layout=True)


	FFT_pos = FFT[1:len(FFT)/2]
	FFT_neg = FFT[(len(FFT)/2) + 1:]
	spec = FFT_pos + FFT_neg[::-1]

	# TODO: show grid, more ticks

	subplots[0].set_xscale('log')
	subplots[0].set_xlim([20, 20000])
	subplots[0].plot(FFT_x[1:len(FFT_x)/2], spec, c='k', lw=.1)
	subplots[0].set_xlabel('frequency (Hz)')

	plot_waveform(subplots[1], sig, embed_crop=crop)


	plt.savefig(out_file)
	plt.close(fig)



def get_fund_freq(sig, expected, window=(1, 2), tol=10):
	samp_freq = 44100.
	window_sec = window
	window = np.array(window) * samp_freq

	sig_crop = sig[int(window[0]): int(window[1])]

	window_len_sec = window_sec[1] - window_sec[0]
	spec_prec = int(100000 / (samp_freq * window_len_sec))  # hz ?

	FFT_x = scipy.fftpack.fftfreq(sig_crop.size * spec_prec, d=1/samp_freq)

	FFT = scipy.fftpack.fft(sig_crop, len(sig_crop) * spec_prec)
	FFT = 20 * scipy.log10(scipy.absolute(FFT)) # convert to db
	FFT_pos = FFT[1:len(FFT)/2]
	FFT_neg = FFT[(len(FFT)/2):]

	if len(FFT_pos) > len(FFT_neg): FFT_pos = FFT_pos[:len(FFT_neg)]	# trim to shorter of two so arrays may be added
	elif len(FFT_pos) < len(FFT_neg): FFT_neg = FFT_neg[:len(FFT_pos)]


	spec = FFT_pos + FFT_neg[::-1]
	spec_x = FFT_x[1:len(FFT_x)/2]

	freq_window_idx = [i for i, x in enumerate(spec_x) if np.abs(expected - x) < tol]
	if len(freq_window_idx) ==0:
		print("ERROR: No fundamental frequency found. Increase 'tol'.")
		sys.exit()
	freq_window_freq = spec_x[freq_window_idx]
	freq_window_amp = spec[freq_window_idx]
	# plot_spec(spec, spec_x, 'lowpass.png')

	max_idx = np.argmax(freq_window_amp)
	fund = freq_window_freq[max_idx]
	return fund
