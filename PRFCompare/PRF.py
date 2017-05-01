from __future__ import division
import time
import subprocess
import os
import itertools
import numpy as np
from os import system, chdir

from sys import platform

from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PersistentHomology.BuildComplex import build_filtration
from DCE.DCETools import embed


def get_filtration(in_filename, params, start=0):
	# lines = open(in_filename).readlines()
	print "building filtration..."
	in_filename = os.getcwd() + '/' + in_filename
	os.chdir('PersistentHomology/')
	filtration = build_filtration(in_filename, params)

	witness_coords = filtration[1][1]
	landmark_coords = filtration[1][0]
	complexes = sorted(list(filtration[0]))
	np.save('temp_data/complexes.npy', complexes)
	np.save('temp_data/witness_coords.npy', witness_coords)
	np.save('temp_data/landmark_coords.npy', landmark_coords)
	return complexes


def get_interval_data():
	""" formats perseus output """
	birth_t, death_t = np.loadtxt('PRFCompare/perseus/perseus_out_1.txt', unpack=True, ndmin=1)

	epsilons = np.loadtxt('PersistentHomology/temp_data/epsilons.txt')
	lim = np.max(epsilons)

	birth_e = []
	death_e = []

	timess = np.vstack([birth_t, death_t]).T
	for times in timess:
		if times[1] != - 1:
			birth_e.append(epsilons[int(times[0])])
			death_e.append(epsilons[int(times[1])])


	immortal_holes = []
	for i, death_time in np.ndenumerate(death_t):    # place immortal holes at [birth time, time lim]
		if death_time == -1:
			immortal_holes.append([epsilons[int(birth_t[i])], lim * .95])
	immortal_holes = np.array(immortal_holes)

	if len(immortal_holes):
		birth_e.extend(immortal_holes[:,0])
		death_e.extend(immortal_holes[:,1])



	try:
		count = np.zeros(len(birth_t))
	except TypeError:
		count = [0]
	for i, pt in enumerate(zip(birth_e, death_e)):
		for scanner_pt in zip(birth_e, death_e):
			if pt == scanner_pt:
				count[i] += 1



	points = np.asarray([birth_e,
						 death_e,
						 count]).T
	points = np.vstack({tuple(row) for row in points})  # toss duplicates


	x, y, z = points[:,0], points[:,1], points[:,2]

	return x, y, z, lim


def get_homology(filt_list):

	""" calls perseus, creating perseus_out_*.tdt """

	# if filt_list == 'read':
	# 	filt_list = np.load('temp_data/complexes.npy')

	def group_by_birth_time(complex_ID_list):
		"""Reformats 1D list of SimplexBirth objects into 2D array of
		landmark_set lists, where 2nd index is  birth time (? see below)"""

		# TODO: ensure that if a time t has no births, the row t is empty/skipped

		complex_ID_array = []  # list of complex_at_t lists
		complex_at_t = []  # list of simplices with same birth_time
		i = 0
		time = 0
		list_length = len(complex_ID_list)
		while i < list_length:
			birth_time = complex_ID_list[i].birth_time
			if birth_time == time:
				complex_at_t.append(complex_ID_list[i].landmark_set)
				i += 1
			else:
				complex_ID_array.append(complex_at_t)
				complex_at_t = []
				time += 1
		return complex_ID_array

	def expand_to_2simplexes(filt_array):
		"""for each k-simplex in filtration array, if k > 2, replaces with the
		component 2-simplexes(i.e. all length-3 subsets of landmark_ID_set) """
		for row in filt_array:
			expanded_row = []
			for landmark_ID_set in row:
				expanded_set = list(itertools.combinations(landmark_ID_set, 3)) \
					if len(landmark_ID_set) > 3 else [list(landmark_ID_set)]
				expanded_row.extend(expanded_set)
			row[:] = expanded_row

	def build_perseus_in_file(filt_array):
		print 'building perseus_in.txt...'
		out_file = open('perseus/perseus_in.txt', 'a')
		out_file.truncate(0)
		out_file.write('1\n')
		for idx, row in enumerate(filt_array):
			for simplex in row:
				#   format for perseus...
				line_str = str(len(simplex) - 1) + ' ' + ' '.join(
					str(ID) for ID in simplex) + ' ' + str(idx + 1) + '\n'
				out_file.write(line_str)
		out_file.close()

	filt_array = group_by_birth_time(filt_list)
	expand_to_2simplexes(filt_array)
	filt_array = np.asarray(filt_array)
	build_perseus_in_file(filt_array)

	print 'calling perseus...'
	os.chdir('perseus')

	if platform == "linux" or platform == "linux2":
		subprocess.call("./perseusLin nmfsimtop perseus_in.txt perseus_out", shell=True)

	elif platform == "darwin":  # macOS
		subprocess.call("./perseusMac nmfsimtop perseus_in.txt perseus_out", shell=True)

	else:   # Windows
		subprocess.call("perseusWin.exe nmfsimtop perseus_in.txt perseus_out", shell=True)

	os.chdir('..')
	os.chdir('..')



def build_rank_func(data):
	""" helper for get_rank_func()"""
	x, y, z, max_lim = data
	min_lim = 0

	div = .05 * max_lim
	x_ = np.arange(min_lim, max_lim, div)
	y_ = np.arange(min_lim, max_lim, div)
	xx, yy = np.meshgrid(x_, y_)

	pts = zip(x, y, z)
	grid_pts = zip(np.nditer(xx), np.nditer(yy))
	grid_vals = np.zeros(len(grid_pts))
	for i, grid_pt in enumerate(grid_pts):
		if grid_pt[0] <= grid_pt[1]:
			for pt in pts:
				if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
					grid_vals[i] += pt[2]
		else:
			grid_vals[i] = np.nan
	grid_vals = np.reshape(grid_vals, xx.shape)

	return [xx, yy, grid_vals, max_lim]


def get_rank_func(filename, filt_params):
	filt = get_filtration(filename, filt_params)
	get_homology(filt)
	intervals = get_interval_data()
	return build_rank_func(intervals)


def plot_dists(i_ref, i_arr, dists, out_filename):
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111)
	ax.plot(i_arr, dists)
	ax.set_xlabel('$tau \quad (samples)$')
	ax.set_ylabel('$distance \quad ({\epsilon}^2 \; \# \; holes)$')
	ax.xaxis.set_ticks(i_arr[::2])
	ax.grid()
	ax.set_ylim(bottom=0)
	ax.set_title('reference tau: ' + str(i_ref))
	plt.savefig(out_filename)
	plt.close(fig)


def PRF_dist_plots(dir, base_filename,out_filename, i_ref, i_arr, filt_params, rebuild_filt=True):
	if rebuild_filt:
		# filename = '{}/{}{}'.format(dir, i_ref, base_filename)
		filename = '{}/{}{}.txt'.format(dir, base_filename, i_ref)
		ref_func = get_rank_func(filename, filt_params)
		funcs = []
		for i in i_arr:
			# filename = '{}/{}{}'.format(dir, i, base_filename)
			filename = '{}/{}{}.txt'.format(dir, base_filename, i)
			print '\n============================================='
			print filename
			print '=============================================\n'
			func = get_rank_func(filename, filt_params)
			funcs.append(func)
		funcs = np.array(funcs)
		np.save('PRFCompare/temp_data/PRFs.npy', funcs)
		np.save('PRFCompare/temp_data/PRF_ref.npy', ref_func)

	funcs = np.load('PRFCompare/temp_data/PRFs.npy')
	ref_func = np.load('PRFCompare/temp_data/PRF_ref.npy')
	box_area = (ref_func[3] / len(ref_func[2])) ** 2
	diffs = np.array([np.subtract(func[2], ref_func[2]) for func in funcs])
	dists = np.array([np.abs(np.nansum(diff)) * box_area  for diff in diffs])
	plot_dists(i_ref, i_arr, dists, out_filename)

from PersistentHomology.PersistencePlotter import add_persistence_plot
from PersistentHomology.FiltrationPlotter import make_movie


def persistence_diagram(filename):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	add_persistence_plot(ax)
	plt.savefig(filename)
	plt.close(fig)


from scipy.signal import butter, lfilter, freqz

def auto_crop(sig, length):

	sig_abs = np.abs(sig)

	# http://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units

	def butter_lowpass(cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False)
		return b, a

	def butter_lowpass_filter(data, cutoff, fs, order=5):
		b, a = butter_lowpass(cutoff, fs, order=order)
		y = lfilter(b, a, data)
		return y



	# Filter requirements.
	order = 1
	fs = 44100
	cutoff = 1

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
	crop = (int(st_arg), int(st_arg + length * 44100))

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





def mean_PRF_dist_plots(
		filename_1, filename_2,
		out_filename,
		filt_params,
		crop_1='auto', 		# sec or 'auto'
		crop_2='auto',
		crop_auto_len=.3, 		# sec
		window_size=.05,		# sec
		num_windows=10,
		mean_samp_num=5,
		wav_samp_rate = 44100, 	# hz
		tau=50,		# samps
		PD_movie_int = 5,
		normalize_volume=True
		):

	def make_movie_and_PD(filename):

		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = 'compare_{:s}_{:d}_'.format(base_name, i)
		PD_filename = 'output/PRFCompare/PDs_and_movies/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/PDs_and_movies/' + comp_name + 'movie.mp4'

		persistence_diagram(PD_filename)

		color_scheme = 'none'
		camera_angle = (135, 55)
		alpha = 1
		dpi = 150
		max_frames = None
		hide_1simplexes = False
		save_frames = False
		framerate = 1

		title_block_info = [filename, 'worm {:d} of {:d}'.format(i, num_windows), filt_params, color_scheme, camera_angle, alpha, dpi, max_frames, hide_1simplexes]
		make_movie(movie_filename, title_block_info, color_scheme, alpha, dpi, framerate, camera_angle, hide_1simplexes, save_frames)

	path = 'output/PRFCompare/PDs_and_movies/'
	old_files = os.listdir(path)
	if old_files and PD_movie_int:
		ans = raw_input('Clear old files in ' + path + ' ? (y/n) \n')
		if ans == 'y':
			for f in old_files:
				if f != '.gitkeep':
					os.remove(path + f)
		else:
			print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'

	filt_params.update({'worm_length' : np.floor(window_size * wav_samp_rate).astype(int)})

	# print 'worm_length:', filt_params['worm_length']

	window_size_samp = int(np.array(window_size) * wav_samp_rate)
	# window_step_samp = int(window_step * wav_samp_rate)

	funcs_1 = []
	sig_1_full = np.loadtxt(filename_1)
	if normalize_volume: sig_1_full = sig_1_full / np.max(sig_1_full)

	if crop_1 == 'auto':
		crop_1_samp = auto_crop(sig_1_full, crop_auto_len)
	else:
		crop_1_samp = np.floor(np.array(crop_1) * wav_samp_rate).astype(int)
	sig_1 = sig_1_full[crop_1_samp[0]:crop_1_samp[1]]


	start_pts = np.floor(np.linspace(0, len(sig_1), num_windows, endpoint=False)).astype(int)
	for i, pt in enumerate(start_pts[:-1]):
		print '\n============================================='
		print filename_1.split('/')[-1], i, pt
		print '=============================================\n'
		window = np.asarray(sig_1[pt:pt + window_size_samp])
		# if normalize_volume: window = window / np.max(window)
		np.savetxt('PRFCompare/temp_data/temp_sig1.txt', window)
		embed('PRFCompare/temp_data/temp_sig1.txt', 'PRFCompare/temp_data/temp_worm1.txt',
			  'none', tau, 2, 4410 )
		func = get_rank_func('PRFCompare/temp_data/temp_worm1.txt', filt_params)
		funcs_1.append(func)

		if PD_movie_int:
			if i % PD_movie_int == 0:
				make_movie_and_PD(filename_1)



	funcs_2 = []
	sig_2_full = np.loadtxt(filename_2)
	if normalize_volume: sig_2_full = sig_2_full / np.max(sig_2_full)

	if crop_2 == 'auto':
		crop_2_samp = auto_crop(sig_2_full, crop_auto_len)
	else:
		crop_2_samp = np.floor(np.array(crop_2) * wav_samp_rate).astype(int)

	sig_2 = sig_2_full[crop_2_samp[0]:crop_2_samp[1]]


	start_pts = np.floor(np.linspace(0, len(sig_2), num_windows, endpoint=False)).astype(int)
	for i, pt in enumerate(start_pts[:-1]):
		print '\n============================================='
		print filename_2.split('/')[-1], i, pt
		print '=============================================\n'
		window = np.asarray(sig_2[pt:pt + window_size_samp])
		# if normalize_volume: window = window /
		np.savetxt('PRFCompare/temp_data/temp_sig2.txt', window,)
		embed('PRFCompare/temp_data/temp_sig2.txt', 'PRFCompare/temp_data/temp_worm2.txt',
			  'none', tau, 2, 4410)
		func = get_rank_func('PRFCompare/temp_data/temp_worm2.txt', filt_params)
		funcs_2.append(func)

		if PD_movie_int:
			if i % PD_movie_int == 0:
				make_movie_and_PD(filename_2)





	funcs_1 = np.load('PRFCompare/temp_data/funcs_1.npy')
	funcs_2 = np.load('PRFCompare/temp_data/funcs_2.npy')

	mean_1_samps = funcs_1[::num_windows//mean_samp_num]
	mean_2_samps = funcs_2[::num_windows//mean_samp_num]

	funcs_1_avg = np.mean(mean_1_samps, axis=0)
	funcs_2_avg = np.mean(mean_2_samps, axis=0)


	# box_area = (funcs_1_avg[3] / len(funcs_1_avg[2])) ** 2
	box_area = 1

	diffs1_vs_1 = np.array([np.subtract(func[2], funcs_1_avg[2]) for func in funcs_1])
	dists1_vs_1 = np.array([np.abs(np.nansum(diff)) * box_area for diff in diffs1_vs_1])

	diffs2_vs_1 = np.array([np.subtract(func[2], funcs_1_avg[2]) for func in funcs_2])
	dists2_vs_1 = np.array([np.abs(np.nansum(diff)) * box_area for diff in diffs2_vs_1])


	diffs1_vs_2 = np.array([np.subtract(func[2], funcs_2_avg[2]) for func in funcs_1])
	dists1_vs_2 = np.array([np.abs(np.nansum(diff)) * box_area for diff in diffs1_vs_2])

	diffs2_vs_2 = np.array([np.subtract(func[2], funcs_2_avg[2]) for func in funcs_2])
	dists2_vs_2 = np.array([np.abs(np.nansum(diff)) * box_area for diff in diffs2_vs_2])

	# x = np.concatenate([start_pts[:-1], start_pts[:-1] + start_pts[-2]])

	from DCE.DCEPlotter import plot_waveform
	def plot_dists(dists1, dists2, title, out_filename):
		fig = plt.figure(figsize=(10, 5))
		ax1 = fig.add_subplot(221)
		ax1.plot(dists1)
		ax1.grid()
		ax1.set_ylim(bottom=0)

		ax2 = fig.add_subplot(222, sharey=ax1)
		ax2.plot(dists2)
		ax2.grid()
		plt.setp(ax2.get_yticklabels(), visible=False)

		ax1.set_title (filename_1.split('/')[-1])
		ax2.set_title (filename_2.split('/')[-1])

		ax3 = fig.add_subplot(223)
		# ax3.plot(sig_1)
		crop = np.asarray(crop_1_samp) / 44100
		plot_waveform(ax3, sig_1_full, crop)

		ax4 = fig.add_subplot(224, sharey=ax3)
		# ax4.plot(sig_2)
		crop = np.asarray(crop_2_samp) / 44100
		plot_waveform(ax4, sig_2_full, crop)


		fig.suptitle(title)

		plt.savefig(out_filename)


		plt.close(fig)

	base_filename = out_filename.split('.')[0]
	ext = out_filename.split('.')[1]


	plot_dists(dists1_vs_1, dists2_vs_1, 'ref mean: left', base_filename + '__ref_left.' + ext)
	plot_dists(dists1_vs_2, dists2_vs_2, 'ref mean: right', base_filename + '__ref_right.' + ext)




def see(filename, filt_params):

	filt = get_filtration(filename, filt_params)
	get_homology(filt)
	intervals = get_interval_data()
	max_lim = intervals[3]
	x, y, z = build_rank_func(intervals)
	z = np.log10(z + 1)

	fig = plt.figure(figsize=(12, 4), dpi = 300, tight_layout=True)
	ax = fig.add_subplot(131, projection='3d')
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.view_init(35, 135)
	ax.plot_surface(x, y, z)

	# plt.show()

	ax = fig.add_subplot(132)
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.set_aspect('equal')
	ax.contourf(x, y, z)

	ax = fig.add_subplot(133)
	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.set_aspect('equal')
	ax.scatter(intervals[0], intervals[1], s=intervals[2] * 5)

	plt.savefig('rank_function_log.png')

