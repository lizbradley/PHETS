''' helpers for miscellaneous tasks eg file management '''

import os, sys, inspect
import subprocess

from memory_profiler import profile


def mem_profile(f, flag):
	if flag: return profile(stream=f)

	else: return lambda x: x


def normalize_volume(sig):
	return np.true_divide(sig, np.max(np.abs(sig)))


def idx_to_freq(idx):
	return np.power(2, (40 - float(idx)) / 12) * 440


def sec_to_samp(crop):
	return (np.array(crop) * WAV_SAMPLE_RATE).astype(int)



# consolidate following three functions

def check_overwrite(out_file_name):
	os.chdir('output')
	if os.path.exists(out_file_name):
		overwrite = raw_input(out_file_name + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			print 'Goodbye'
			sys.exit()
	os.chdir('..')


def clear_old_files(path, see_samples):
	old_files = os.listdir(path)
	if old_files and see_samples:
		ans = raw_input('Overwrite files in ' + path + '? (y/n/q) \n')
		if ans == 'y':
			for f in old_files:
				if f != '.gitkeep':
					if not path.endswith('/'):
						path = path + '/'
					os.remove(path + f)
		elif ans == 'q':
			print 'Goodbye'
			sys.exit()
		else:
			print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'


def clear_dir(dir):

	files = os.listdir(dir)
	if files:
		r = raw_input('Clear files in {}? (y/n/q) '.format(dir))
	else:
		return True

	if r == 'y':
		for f in files:
			os.remove(dir + f)
		return True

	elif r == 'n':
		return False

	else:
		print 'Goodbye'
		sys.exit()


def clear_temp_files(dir):
	if not dir.endswith('/'):
		dir = dir + '/'
	for f in os.listdir(dir):
		if f != '.gitignore':
			os.remove(dir + f)


def lambda_to_str(f):
	return inspect.getsourcelines(f)[0][0].split(':')[1]

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__


def print_title(str):
	if str:
		print '=================================================================='
		print str
		print '=================================================================='


def count_lines(dir, blanks=True):

	def file_len(fname):
		count = 0
		with open(fname) as f:
			for line in f:
				if line != '\n' or blanks:
					count += 1
		return count

	count_f = 0
	count_l = 0
	for root, dirs, files in os.walk(dir, topdown=False):
		for name in files:
			if name.endswith('.py') and not name.endswith('Tester.py'):
				fname = os.path.join(root, name)
				length = file_len(fname)

				count_f += 1
				count_l += length

				print '{}: {}'.format(fname, length)

	print_title('num files: {}\tnum lines: {}'.format(count_f, count_l))


def remove_old_frames(dir):
	for f in os.listdir(dir):
		if f.endswith(".png"):
			os.remove(dir + f)


def remove_files_by_type(dir, ftype):
	for f in os.listdir(dir):
		if f.endswith(ftype):
			os.remove(dir + f)
			print 'removed {}'.format(dir + f)


def frames_to_movie(out_filename, frame_path, framerate=1, loglevel='panic'):

	if os.path.exists(out_filename):
		overwrite = raw_input(out_filename + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			sys.exit()

	sys.stdout.write('\rconsolidating frames...')
	sys.stdout.flush()
	cmd = [
		'ffmpeg',
		'-loglevel', loglevel,
		'-y',
		'-framerate', str(framerate),
		'-i', frame_path,
		'-r', str(24),
		# '-aspect', '{}:{}'.format(aspect[0], aspect[1]),
		out_filename

	]
	subprocess.call(cmd)
	print 'done.'
	print 'see {}'.format(out_filename)


# BELOW ARE MIGRATED FROM DCE MODULE, NEED CLEANUP #


import sys
import os
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as pyplot
import math
from scipy.signal import butter, lfilter, freqz

from config import WAV_SAMPLE_RATE


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


