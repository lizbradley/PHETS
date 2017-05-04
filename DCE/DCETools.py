import sys
import os
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as pyplot
import math
from scipy.signal import butter, lfilter, freqz


WAV_SAMPLE_RATE = 44100

def wav_to_txt(wav_file_name, output_file_name, crop=(0, 1)):
	print "converting wav to txt..."
	sampFreq, sig = wavfile.read(wav_file_name)
	bounds = len(sig) * np.array(crop)
	sig = sig[int(bounds[0]):int(bounds[1])]
	np.savetxt(output_file_name, sig)


def embed(
		input_file_name, output_file_name,
		embed_crop,		# sec
		tau,			# samples
		m,
		ds_rate=1,
		channel=0
	):


	if isinstance(input_file_name, basestring):
		input_file = open(input_file_name, "r")
		lines = input_file.read().split("\n")
		input_file.close()

		# apply bounds #########################################
		worm_length_sec = len(lines) / WAV_SAMPLE_RATE
		if isinstance(embed_crop, basestring):
			if embed_crop == 'none': embed_crop_norm = [0, 1]
		else:
			embed_crop_norm = [float(t) / worm_length_sec for t in embed_crop]
		bounds = len(lines) * np.array(embed_crop_norm)
		lines = lines[int(bounds[0]): int(bounds[1]): ds_rate]
		########################################################

		series = []
		for line in lines:
			if line != "":
				channels = [x for x in line.split(" ") if x != ""]
				series.append(float(channels[channel]))

		end = len(lines) - (tau * (m - 1)) - 1


	else:
		series = input_file_name			# accept array instead of filename
		embed_crop_samp = np.array(embed_crop) * WAV_SAMPLE_RATE
		series = series[int(embed_crop_samp[0]):int(embed_crop_samp[1])]
		end = len(series) - (tau * (m - 1)) - 1

	output_file = open(output_file_name, "w")
	output_file.truncate(0)

	for i in xrange(end):
		for j in xrange(m):
			output_file.write("%f " % series[i + (j*tau)])
		if i < end:
			output_file.write("\n")
	output_file.close()


def auto_crop(sig, length):
	""" 
	finds max of volume envelope: (xmax, ymax)
	get first point (x, y) on envelope where y < .1 * ymax and x > xmax """

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

	sig_abs = np.abs(sig)
	order, fs, cutoff = 1, WAV_SAMPLE_RATE, 1		# filter params
	envelope = butter_lowpass_filter(sig_abs, cutoff, fs, order)

	n = len(sig)
	T = n/fs
	t = np.linspace(0, T, n, endpoint=False)

	max_arg = np.argmax(envelope)
	max = envelope[max_arg]

	st_arg = 0
	for i in xrange(max_arg, len(envelope)):
		if envelope[i] < .1 * max:
			st_arg = i
			break
	print 'crop start:', t[st_arg]
	crop = (int(st_arg), int(st_arg + length * WAV_SAMPLE_RATE))

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(t, sig, label='data', color='lightblue')
	# ax.plot(t, envelope, linewidth=1, label='filtered data', color='red')
	# ax.set_xlabel('Time [sec]')
	# ax.grid()
	# ax.legend()
	#
	# ax.plot(t[max_arg], envelope[max_arg], color='k', marker='.')
	# print t[max_arg], envelope[max_arg]
	#
	# ax.plot(t[st_arg], envelope[st_arg], color='k', marker='.')
	#
	# plt.show()

	return crop


def auto_tau(tau_cmd, filename, note_index, tau_T, crop):
	""" 
		helper for mean_PRF_dist_plots(). would be nice to use in DCEMovies.compare_multi() as well
		returns tau in samples
	"""
	if isinstance(tau_cmd, basestring):
		if not note_index:
			print "ERROR: 'note_index' required for"

		fname_index = int(filename.split('/')[-1].split('-')[0])
		if fname_index != note_index:
			r = input("WARNING: 'note_index' does not seem to match fname_index. 'note_index' is required for auto tau functionality. Continue? (y/n)")
			if r == 'n':
				sys.exit()
		ideal_freq = math.pow(2, (40 - float(note_index)) / 12) * 440  # Hz, descending index

	if not isinstance(tau_cmd, basestring):
		tau = tau_cmd * WAV_SAMPLE_RATE
	
	elif tau_cmd == 'auto detect':
		f = get_fund_freq(filename, ideal_freq, crop)
		T = 1. / f
		tau_sec = tau_T * T
		tau = tau_sec * WAV_SAMPLE_RATE

	elif tau_cmd == 'auto ideal':
		f = ideal_freq
		T = 1./f
		tau_sec = tau_T * T
		tau = tau_sec * WAV_SAMPLE_RATE
			
	else:
		print 'ERROR: tau_cmd not recognized.'
		sys.exit()

	print 'using tau:', tau
	return tau




def auto_embed(
		filename,

		crop='auto',
		crop_auto_len=.3,

		tau='auto ideal',
		tau_T = np.pi,
		note_index=None
		):

	if isinstance(crop, basestring):
		if crop == 'auto':
			crop = np.array(auto_crop(np.loadtxt(filename), crop_auto_len)) / float(WAV_SAMPLE_RATE)
		else:
			print "ERROR: embed_crop_1 not recognized. Use 'auto' or explicit (seconds)."
			sys.exit()



	if isinstance(tau, basestring):
		if not note_index: print "ERROR: note index required for tau='auto ideal' and tau='auto detect'"

		ideal_freq = math.pow(2, (40 - float(note_index)) / 12) * 440  # Hz, descending index

		if tau == 'auto detect': f = get_fund_freq(filename, ideal_freq, window=crop)

		if tau == 'auto ideal': f = ideal_freq

		else:
			print 'ERROR: tau not recognized.'
			sys.exit()

	embed(filename, 'DCE/temp_data/embedded_coords_comp1.txt', crop_1, tau_1, m, ds_rate=ds_rate)



def rename_files():
	os.chdir('..')
	os.chdir('datasets/time_series/ivy162_piano/a440')
	raw_input(os.getcwd())
	[os.rename(f, f.replace('49', '40')) for f in os.listdir('.') if f.endswith('.wav') or f.endswith('.txt')]


def rename_files_shift_index():
	os.chdir('input/')
	for f in os.listdir('.'):
		i_in = int(f.split('-')[0])
		base = f.split('-')[1]
		i_out = i_in + 1
		os.rename(f, "temp{:02d}-{}".format(i_out, base))

	for f in os.listdir('.'):
		if 'temp' in f:
			os.rename(f, f.replace('temp', ''))



def batch_wav_to_txt(dir_name):
	os.chdir(dir_name)
	[wav_to_txt(f, f.replace('.wav', '.txt')) for f in os.listdir('.') if f.endswith('.wav')]



def get_fund_freq(filename, expected, window=(1, 2), tol=10):
	samp_freq = 44100.
	window_sec = window
	window = np.array(window) * samp_freq
	sig = np.loadtxt(filename)

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

# from scipy.interpolate import interp1d

def plot_spec(spec_x, spec, out_file):
	fig = pyplot.figure()
	plt = fig.add_subplot(111)
	plt.set_xscale('log')
	plt.set_xlim([20, 20000])
	plt.plot(spec, spec_x, c='k', lw=.1)
	plt.set_xlabel('frequency (Hz)')
	plt.grid()


	pyplot.savefig(out_file)
	pyplot.close(fig)


def plot_power_spectrum(sig, out_file, crop=(1,2)):
	from DCEPlotter import plot_waveform
	samp_freq = 44100.

	if crop != 'none':
		window = np.array(crop) * samp_freq
		sig_crop = sig[int(window[0]):int(window[1])]
	else:
		sig_crop = sig
	FFT = scipy.fftpack.fft(sig_crop)
	FFT = 20 * scipy.log10(scipy.absolute(FFT)) # convert to db
	FFT_x = scipy.fftpack.fftfreq(len(FFT), d=1 / samp_freq)

	fig, subplots = pyplot.subplots(2, figsize=(6, 3), dpi=300, tight_layout=True)


	FFT_pos = FFT[1:len(FFT)/2]
	FFT_neg = FFT[(len(FFT)/2) + 1:]
	spec = FFT_pos + FFT_neg[::-1]

	# TODO: show grid, more ticks

	subplots[0].set_xscale('log')
	subplots[0].set_xlim([20, 20000])
	subplots[0].plot(FFT_x[1:len(FFT_x)/2], spec, c='k', lw=.1)
	subplots[0].set_xlabel('frequency (Hz)')

	plot_waveform(subplots[1], sig, embed_crop=crop)


	pyplot.savefig(out_file)
	pyplot.close(fig)

def batch_flac_to_wav(dir):
	import soundfile as sf
	os.chdir(dir)
	for f in os.listdir('.'):
		if f.endswith('.flac'):
			out_file = f.split('.')[0] + '.wav'
			data, samplerate = sf.read(f)
			sf.write(out_file, data, samplerate)
			os.remove(f)


if __name__ == '__main__':
	rename_files()
	# rename_files_shift_index()
	# batch_wav_to_txt('C:\Users\PROGRAMMING\Documents\CU_research\piano_data\C134C')
	# batch_wav_to_txt('input/viol_data')
	# get_fund_freq('input/viol_data/01-viol.txt', window=(1, 2))
	# get_fund_freq('input/piano_data/C134C/24-C134C.txt', window=(1, 2))
	# os.chdir('..')
	print os.getcwd()

	# dir = '../datasets/time_series/clarinet/sustained/high_quality'
	# batch_flac_to_wav(dir)
	# batch_wav_to_txt(dir)
