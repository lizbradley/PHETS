from __future__ import division
import time
import subprocess
import os
import sys
from sys import platform
import itertools
import numpy as np
from os import system, chdir

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DCE.DCE import embed
from DCE.Plotter import plot_waveform
from DCE.Tools import auto_tau

from PersistentHomology.BuildComplex import build_filtration
from PersistentHomology.PersistencePlotter import add_persistence_plot
from PersistentHomology.FiltrationPlotter import make_movie

from DCE.Tools import auto_crop

WAV_SAMPLE_RATE = 44100.

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
	# NOTE: should be merged back into PersistencePlotter
	birth_t, death_t = np.loadtxt('PersistentHomology/perseus/perseus_out_1.txt', unpack=True, ndmin=1)

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
	except TypeError:	# only one interval point
		count = [0]
	for i, pt in enumerate(zip(birth_e, death_e)):
		for scanner_pt in zip(birth_e, death_e):
			if pt == scanner_pt:
				count[i] += 1

	points = np.asarray([birth_e, death_e, count]).T
	points = np.vstack({tuple(row) for row in points})  # toss duplicates

	x, y, z = points[:,0], points[:,1], points[:,2]

	return x, y, z, lim


def get_homology(filt_list):
	""" calls perseus, creating perseus_out_*.txt
		TODO: move to PersistentHomology and replace equivalent code there
	"""

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


def build_PRF(data):
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


def get_PRF(filename, filt_params):
	filt = get_filtration(filename, filt_params)
	get_homology(filt)
	intervals = get_interval_data()
	f = build_PRF(intervals)
	return f




def persistence_diagram(filename):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	add_persistence_plot(ax)
	plt.savefig(filename)
	plt.close(fig)


def PRF_contour_plot(ax, func):

	x, y, z, max_lim = func
	# z = np.log10(z + 1)

	ax.set_xlim([0, max_lim])
	ax.set_ylim([0, max_lim])
	ax.set_aspect('equal')
	ax.contourf(x, y, z)






def PRF_dist_plot(dir, base_filename, fname_format,
				  out_filename,
				  i_ref, i_arr,
				  filt_params,
				  PD_movie_int=5):

	""" plots distance from reference rank function over a range of embedded input files"""

	def plot_distances(i_ref, i_arr, dists, out_filename):
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(111)
		ax.plot(i_arr, dists)
		ax.axvline(x=i_ref, linestyle='--', color='k')
		ax.set_xlabel('$tau \quad (samples)$')
		# ax.set_ylabel('$distance \quad ({\epsilon}^2 \; \# \; holes)$')
		ax.set_ylabel('$distance$')
		ax.xaxis.set_ticks(i_arr[::2])
		ax.grid()
		ax.set_ylim(bottom=0)
		title = ax.set_title(base_filename + ' PRF distances')
		title.set_position([.5, 1.05])
		plt.savefig(out_filename)
		plt.close(fig)

	def get_in_filename(i):
		if fname_format == 'i base':
			filename = '{}/{}{}'.format(dir, i, base_filename)
		elif fname_format == 'base i':
			filename = '{}/{}{}.txt'.format(dir, base_filename, i)
		else:
			print "ERROR: invalid fname_format. Valid options: 'i base', 'base i'"
			sys.exit()
		return filename

	def make_movie_and_PD(filename, i, ref=False):
		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = 'ref_compare_{}_{}_'.format(base_name, i)
		if ref: comp_name += 'REFERENCE'
		PD_filename = 'output/PRFCompare/PDs_and_movies/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/PDs_and_movies/' + comp_name + 'movie.mp4'

		color_scheme = 'none'
		camera_angle = (135, 55)
		alpha = 1
		dpi = 150
		max_frames = None
		hide_1simplexes = False
		save_frames = False
		framerate = 1
		title_block_info = [filename, '', filt_params, color_scheme, camera_angle, alpha, dpi, max_frames, hide_1simplexes]

		persistence_diagram(PD_filename)
		make_movie(movie_filename, title_block_info, color_scheme, alpha, dpi, framerate, camera_angle, hide_1simplexes, save_frames)

	filename = get_in_filename(i_ref)

	ref_func = get_PRF(filename, filt_params)


	# PRF_contour_plot(ref_func, 'output/PRFCompare/REFERENCE_CONTOUR.png')
	# fig = plt.figure()
	ax = plt.add_subplot(111)
	PRF_contour_plot(ax, ref_func)
	plt.savefig('output/PRFCompare/ref_contour.png')

	if PD_movie_int: make_movie_and_PD(filename, i_ref, ref=True)
	
	funcs = []
	for i in i_arr:
		filename = get_in_filename(i)
		print '\n=================================================='
		print filename
		print '==================================================\n'
		func = get_PRF(filename, filt_params)
		funcs.append(func)

		if PD_movie_int:
			if i % PD_movie_int == 0:
				make_movie_and_PD(filename, i)

	funcs_z = np.asarray(funcs)[:,2]
	ref_func_z = ref_func[2]


	# box_area = (ref_func[3] / len(ref_func[2])) ** 2
	box_area = 1

	diffs = np.array([np.subtract(func_z, ref_func_z) for func_z in funcs_z])

	dists = np.array([np.nansum(np.abs(diff)) * box_area for diff in diffs])

	plot_distances(i_ref, i_arr, dists, out_filename)



def mean_PRF_dist_plots(
		filename_1, filename_2,
		out_filename,
		filt_params,

		crop_1='auto', 						# sec or 'auto'
		crop_2='auto',
		auto_crop_length=.3, 					# sec

		window_size=.05,					# sec
		num_windows=10,						# per file
		mean_samp_num=5,					# per file

		tau=.001,							# sec or 'auto ideal' or 'auto detect'
		tau_T=np.pi,
		note_index=None,					#

		normalize_volume=True,

		PD_movie_int = 5,
	):


	def clear_old_files():
		path = 'output/PRFCompare/PDs_and_movies/'
		old_files = os.listdir(path)
		if old_files and PD_movie_int:
			ans = raw_input('Clear old files in ' + path + '? (y/n) \n')
			if ans == 'y':
				for f in old_files:
					if f != '.gitkeep':
						os.remove(path + f)
			else:
				print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'


	def make_movie_and_PD(filename, i, ref=False):

		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = 'mean_compare_{:s}_{:d}_'.format(base_name, i)
		if ref: comp_name += 'MEAN'
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


	def get_PRFs(filename, crop_cmd, tau_cmd):

		sig_full = np.loadtxt(filename)
		if normalize_volume: sig_full = sig_full / np.max(sig_full)

		crop = auto_crop(crop_cmd, sig_full, auto_crop_length)		# returns crop in seconds
		sig = sig_full[int(crop[0] * WAV_SAMPLE_RATE) : int(crop[1] * WAV_SAMPLE_RATE)]

		f_deal, f_disp, tau = auto_tau(tau_cmd, sig, note_index, tau_T, crop, filename)

		funcs = []
		start_pts = np.floor(np.linspace(0, len(sig), num_windows, endpoint=False)).astype(int)
		for i, pt in enumerate(start_pts[:-1]):

			print '\n============================================='
			print filename.split('/')[-1], 'worm #', i
			print '=============================================\n'

			window = np.asarray(sig[pt:pt + window_size_samp])
			embed(window, 'PRFCompare/temp_data/temp_worm.txt',
				  False, tau, 2)

			func = get_PRF('PRFCompare/temp_data/temp_worm.txt', filt_params)
			funcs.append(func)	# select grid_vals (third element)

			if PD_movie_int:
				if i % PD_movie_int == 0:
					pass
					make_movie_and_PD(filename, i)

		return crop, sig_full, funcs


	def plot_distances(d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2, out_filename):
		fig = plt.figure(figsize=(14, 6), tight_layout=True)

		ax1 = fig.add_subplot(321)
		ax1.plot(d_1_vs_1)
		ax1.axhline(y=np.mean(d_1_vs_1), linestyle='--', color='k')
		ax1.grid()
		ax1.set_ylim(bottom=0)
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.setp(ax1.get_xticklines(), visible=False)

		ax2 = fig.add_subplot(322, sharey=ax1)
		ax2.plot(d_2_vs_1)
		ax2.axhline(y=np.mean(d_2_vs_1), linestyle='--', color='k')

		ax2.grid()
		plt.setp(ax2.get_yticklabels(), visible=False)
		plt.setp(ax2.get_yticklines(), visible=False)
		plt.setp(ax2.get_xticklabels(), visible=False)
		plt.setp(ax2.get_xticklines(), visible=False)


		ax3 = fig.add_subplot(323, sharey=ax1)
		ax3.plot(d_1_vs_2)
		ax3.axhline(y=np.mean(d_1_vs_2), linestyle='--', color='k')
		ax3.grid()

		ax4 = fig.add_subplot(324, sharey=ax1)
		ax4.plot(d_2_vs_2)
		ax4.axhline(y=np.mean(d_2_vs_2), linestyle='--', color='k')

		ax4.grid()
		plt.setp(ax4.get_yticklabels(), visible=False)
		plt.setp(ax4.get_yticklines(), visible=False)


		ax1.set_title (filename_1.split('/')[-1])
		ax2.set_title (filename_2.split('/')[-1])
		del_12 = np.abs(np.mean(d_1_vs_1) - np.mean(d_1_vs_2))
		del_34 = np.abs(np.mean(d_2_vs_1) - np.mean(d_2_vs_2))

		ax1.set_ylabel('ref: left \n \n $\Delta$: {:.2f}'.format(del_12), rotation=0, size='large', labelpad=50)
		ax3.set_ylabel('ref: right \n \n $\Delta$: {:.2f}'.format(del_34), rotation=0, size='large', labelpad=50)

		ax5 = fig.add_subplot(325)
		plot_waveform(ax5, sig_1_full, crop_1)

		ax6 = fig.add_subplot(326, sharey=ax5)
		plot_waveform(ax6, sig_2_full, crop_2)
		plt.setp(ax6.get_yticklabels(), visible=False)
		plt.setp(ax6.get_yticklines(), visible=False)


		plt.savefig(out_filename)


		plt.close(fig)


	# ======================= SETUP	============================
	clear_old_files()
	filt_params.update({'worm_length' : np.floor(window_size * WAV_SAMPLE_RATE).astype(int)})
	print 'using worm_length:', filt_params['worm_length']
	window_size_samp = int(window_size * WAV_SAMPLE_RATE)

	crop_1_cmd = crop_1
	crop_2_cmd = crop_2

	# ===========================================================

	crop_1, sig_1_full, funcs_1 = get_PRFs(filename_1, crop_1_cmd, tau)
	crop_2, sig_2_full, funcs_2 = get_PRFs(filename_2, crop_2_cmd, tau)

	if not np.array_equal(funcs_1[0][0], funcs_1[1][0]):
		print 'ERROR: variable PRF domain detected. mean_PRF_dist_plot() may only be used with explicit max_filtartion_param.'
		sys.exit()

	funcs_1_z = np.asarray(funcs_1)[:, 2]
	funcs_2_z = np.asarray(funcs_2)[:, 2]

	mean_1_funcs_z = funcs_1_z[::num_windows//mean_samp_num]
	mean_2_funcs_z = funcs_2_z[::num_windows//mean_samp_num]

	funcs_1_avg_z = np.mean(mean_1_funcs_z, axis=0)
	funcs_2_avg_z = np.mean(mean_2_funcs_z, axis=0)


	ref_func_1 = funcs_1[0]			# get xx, yy
	ref_func_1[2] = funcs_1_avg_z

	ref_func_2 = funcs_2[0]			# get xx, yy
	ref_func_2[2] = funcs_2_avg_z

	fig = plt.figure(tight_layout=True, figsize=(8, 4))
	ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
	PRF_contour_plot(ax1, ref_func_1)
	PRF_contour_plot(ax2, ref_func_2)
	ax1.set_title('left')
	ax2.set_title('right')
	fig.savefig('output/PRFCompare/mean_PRF_REF_contour_plots.png')

	box_area = 1

	diffs1_vs_1 = np.array([np.subtract(func, funcs_1_avg_z) for func in funcs_1_z])
	dists1_vs_1 = np.array([np.nansum(np.abs(diff)) * box_area for diff in diffs1_vs_1])

	diffs2_vs_1 = np.array([np.subtract(func, funcs_1_avg_z) for func in funcs_2_z])
	dists2_vs_1 = np.array([np.nansum(np.abs(diff)) * box_area for diff in diffs2_vs_1])

	diffs1_vs_2 = np.array([np.subtract(func, funcs_2_avg_z) for func in funcs_1_z])
	dists1_vs_2 = np.array([np.nansum(np.abs(diff)) * box_area for diff in diffs1_vs_2])

	diffs2_vs_2 = np.array([np.subtract(func, funcs_2_avg_z) for func in funcs_2_z])
	dists2_vs_2 = np.array([np.nansum(np.abs(diff)) * box_area for diff in diffs2_vs_2])


	plot_distances(dists1_vs_1, dists2_vs_1, dists1_vs_2, dists2_vs_2, out_filename)



