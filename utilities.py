''' helpers for miscellaneous tasks eg file management '''

import inspect
import subprocess
import sys
import os
import time
import numpy as np
import datetime
from scipy.io import wavfile

from config import SAMPLE_RATE


label = '';

def generate_label(label_str):
    global label
    label = label_str;

def get_label():
    
    global label 
    if label == '':
        generate_label(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"));
    return label;


def get_prfstats_path(out_filename):
    
    out_dir, out_filename = os.path.split(out_filename)
    suffix = get_label();
    out_dir = out_dir + '/' + suffix
    make_dir(out_dir)
    out_filename = out_dir + '/' + out_filename

    return out_dir, out_filename;

def get_prfstats_filts_path(second_suffix=''):
    
    suffix = get_label();
    base = 'prfstats/data/' + suffix;

    try:
	os.makedirs(base);
    except OSError:
	pass

    return (base + '/filts{}.npy'.format(second_suffix))


def idx_to_freq(idx):
	return np.power(2, (40 - float(idx)) / 12) * 440


def sec_to_samp(crop):
	return (np.array(crop) * SAMPLE_RATE).astype(int)


def make_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def clear_dir(dir):
	files = os.listdir(dir)
	if files and not (len(files) == 1 and files[0] in ('.gitignore', '.gitkeep')):
		r = raw_input('Clear files in {}? (y/n/q) '.format(dir))
	else:
		return True

	if r == 'y':
		for f in files:
			if not dir.endswith('/'):
				dir = dir + '/'
			os.remove(dir + f)
		return True

	elif r == 'n':
		return False

	else:
		print 'goodbye'
		sys.exit()


def clear_temp_files(dir):
	if not dir.endswith('/'):
		dir = dir + '/'
	for f in os.listdir(dir):
		if f not in ('.gitignore', '.gitkeep'):
			os.remove(dir + f)


def lambda_to_str(f):
	return inspect.getsourcelines(f)[0][0].split(':')[1]


def block_print():
	sys.stdout = open(os.devnull, 'w')


def enable_print():
	sys.stdout = sys.__stdout__


def print_title(str):
	if str:
		print '=================================================================='
		print str
		print '=================================================================='


def print_still(str):
	sys.stdout.write('\r{}\t\t'.format(str))
	sys.stdout.flush()


def count_lines(dir, blanks=False):

	def file_len(fname):
		count = 0
		with open(fname) as f:
			for line in f:
				if line != '\n' or blanks:
					count += 1
		return count

	def conditions(path, file):
		return (
			file.endswith('.py'),
			not name.endswith('Tester.py'),
			'unit_tests' not in path,
			'docs' not in path,
		)

	count_f = 0
	count_l = 0
	for root, dirs, files in os.walk(dir, topdown=False):
		for name in files:
			if all(conditions(root, name)):
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


def frames_to_movie(out_filename, frame_path, framerate=1, loglevel='error'):
	if os.path.exists(out_filename):
		overwrite = raw_input(out_filename + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			print 'goodbye'
			sys.exit()

	sys.stdout.write('\rconsolidating frames...')
	sys.stdout.flush()
	cmd = [
		'ffmpeg',
		'-loglevel', loglevel,
		'-y',
		'-framerate', str(framerate),
		'-i', frame_path,
		'-r', str(16),
		# '-aspect', '{}:{}'.format(aspect[0], aspect[1]),
		out_filename

	]
	subprocess.call(cmd)
	print 'done.'


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
	[os.rename(f, f.replace('49', '40'))
	 for f in os.listdir('.')
	 if f.endswith('.wav') or f.endswith('.txt')]


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
	[wav_to_txt(f, f.replace('.wav', '.txt'))
	 for f in os.listdir('.')
	 if f.endswith('.wav')]



def timeit(f):
	def timed(*args, **kw):
		ts = time.time()
		result = f(*args, **kw)
		te = time.time()
		print '{} time elapsed: {:.6f}s'.format(f.__name__, (te - ts))
		return result
	return timed
