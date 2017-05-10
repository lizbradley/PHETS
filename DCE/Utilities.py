import sys
import os
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as pyplot
import math
from scipy.signal import butter, lfilter, freqz

WAV_SAMPLE_RATE = 44100

def pwd():
	print os.getcwd()

def wav_to_txt(wav_file_name, output_file_name, crop=(0, 1)):
	print "converting wav to txt..."
	sampFreq, sig = wavfile.read(wav_file_name)
	bounds = len(sig) * np.array(crop)
	sig = sig[int(bounds[0]):int(bounds[1])]
	np.savetxt(output_file_name, sig)





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


def batch_flac_to_wav(dir):
	import soundfile as sf
	os.chdir(dir)
	for f in os.listdir('.'):
		if f.endswith('.flac'):
			out_file = f.split('.')[0] + '.wav'
			data, samplerate = sf.read(f)
			sf.write(out_file, data, samplerate)
			os.remove(f)
