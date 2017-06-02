from __future__ import division
import time
import subprocess
import os
import sys
from sys import platform
import itertools
import numpy as np
from os import system, chdir

from math import ceil


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DCE.DCE import embed
from DCE.Plotter import plot_waveform, plot_waveform_zoom
from DCE.Tools import auto_tau

from PH.Data import Filtration
from PH.Plots import make_PD, make_PRF_plot
from PH.FiltrationMovie import make_movie

from DCE.Tools import auto_crop

WAV_SAMPLE_RATE = 44100.


def clear_old_files(path, PD_movie_int):
	old_files = os.listdir(path)
	if old_files and PD_movie_int:
		ans = raw_input('Clear old files in ' + path + '? (y/n) \n')
		if ans == 'y':
			for f in old_files:
				if f != '.gitkeep':
					os.remove(path + f)
		else:
			print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'


def get_scaled_dists(funcs_z, ref_func_z, weighting_func, metric, scale, PRF_res):

	box_area = (1 / PRF_res) ** 2
	norm_x, norm_y = np.meshgrid(np.linspace(0, 1, PRF_res), np.linspace(0, 1, PRF_res))
	weighting_func_arr = weighting_func(norm_x, norm_y)

	def get_dists(funcs_z, ref_func_z):
		diffs = np.asarray([np.subtract(func_z, ref_func_z) for func_z in funcs_z])
		diffs_weighted = np.asarray([np.multiply(diff, weighting_func_arr) for diff in diffs])
		if metric == 'L1':
			dists = np.asarray([(np.nansum(np.abs(diff))) * box_area for diff in diffs_weighted])
		elif metric == 'L2':
			dists = np.asarray([np.nansum(np.sqrt(np.power(diff, 2))) * box_area for diff in diffs_weighted])
		else:
			print "ERROR: metric not recognized. Use 'L1' or 'L2'."
			sys.exit()
		return dists


	dists = get_dists(funcs_z, ref_func_z)

	dists_0 = get_dists(funcs_z, np.zeros_like(ref_func_z))

	dist_ref_0 = get_dists([ref_func_z], np.zeros_like(ref_func_z))

	dists_ref_0 = [dist_ref_0[0] for i in dists]

	if scale == 'none':
		scaled_dists = dists

	elif scale == 'a':
		scaled_dists = np.true_divide(dists, dists_0)

	elif scale == 'b':
		scaled_dists = np.true_divide(dists, dists_ref_0)

	elif scale == 'a + b':
		scaled_dists = np.true_divide(dists, np.add(dists_0, dists_ref_0))

	else:
		print "ERROR: dist_scale '" + scale + "' is not recognized. Use 'none', 'a', 'b', or 'a + b'."

	return scaled_dists








def PRF_dist_plot(
		dir, base_filename,
		fname_format,
		out_filename,
		filt_params,

		i_ref=15,
		i_arr=np.arange(10, 20, 1),


		weight_func=lambda i, j: 1,

		metric='L2',						# 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',					# 'none', 'a', or 'a + b'

		PRF_res=50,  						# number of divisions used for PRF


		PD_movie_int=5,
):

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

	def make_movie_and_PD(filt, i, ref=False):
		base_name = base_filename.split('/')[-1].split('.')[0]
		comp_name = '{:s}_{:d}_'.format(base_name, i)
		if ref: comp_name += '_REFERENCE_'
		PD_filename = 'output/PRFCompare/ref_PRFC/PDs_and_movies/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/ref_PRFC/PDs_and_movies/' + comp_name + 'movie.mp4'

		make_PD(filt, PD_filename)
		make_movie(filt, movie_filename)

	def get_PRFs():
		funcs = []
		for i in i_arr:
			filename = get_in_filename(i)
			print '\n=================================================='
			print filename
			print '==================================================\n'
			filt = Filtration(filename, filt_params)
			func = filt.get_PRF(PRF_res)
			funcs.append(func)

			if PD_movie_int:
				if i % PD_movie_int == 0:
					make_movie_and_PD(filt, i)


		return np.asarray(funcs)

	# ===================================================== #
	# 				MAIN:	PRF_dist_plots()				#
	# ===================================================== #

	clear_old_files('output/PRFCompare/ref_PRFC/PDs_and_movies/', PD_movie_int)
	ref_filename = get_in_filename(i_ref)
	ref_filt = Filtration(ref_filename, filt_params)
	ref_func = ref_filt.get_PRF(PRF_res)

	if PD_movie_int: make_movie_and_PD(ref_filt, i_ref, ref=True)		# call after get_PRF for reference (loads saved filtration)

	funcs = get_PRFs()		# also makes PDs and movies

	## plot ref PRF ##

	make_PRF_plot(
		ref_func,
	  	'output/PRFCompare/ref_PRFC/PRF_REFERENCE.png',
		params=filt_params,
		in_filename='REF'
	)

	## debugging ##
	# np.save('ref_func_debug.npy', ref_func)
	# np.save('funcs_debug.npy', np.asarray(funcs))
	# ref_func = np.load('ref_func_debug.npy')
	# funcs = np.load('funcs_debug.npy')

	funcs_z = funcs[:,2]

	ref_func_z = ref_func[2]

	dists = get_scaled_dists(funcs_z, ref_func_z, weight_func, metric, dist_scale, PRF_res)

	plot_distances(i_ref, i_arr, dists, out_filename)



def mean_PRF_dist_plots(
		filename_1, filename_2,
		out_filename,
		filt_params,

		load_saved_filtrations=False,

		crop_1='auto', 						# sec or 'auto'
		crop_2='auto',
		auto_crop_length=.3, 				# sec

		window_size=.05,					# sec
		num_windows=10,						# per file
		mean_samp_num=5,					# per file

		tau_1=.001,							# sec or 'auto ideal' or 'auto detect'
		tau_2=.001,
		tau_T=np.pi,
		note_index=None,					#

		normalize_volume=True,

		PRF_res=50,  						# number of divisions used for PRF
		dist_scale='none',					# 'none', 'a', or 'a + b'
		metric='L2',						# 'L1' (abs) or 'L2' (euclidean)
		weight_func=lambda i, j: 1,

		PD_movie_int=5

		):


	def make_movie_and_PD(filt, i, filename):
		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = '{:s}_{:d}_'.format(base_name, i)
		PD_filename = 'output/PRFCompare/mean_PRFC/PDs_and_movies/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/mean_PRFC/PDs_and_movies/' + comp_name + 'movie.mp4'

		make_PD(filt, PD_filename)
		make_movie(filt, movie_filename)

	def crop_sig(sig_full, crop_cmd, auto_crop_len):
		crop = auto_crop(crop_cmd, sig_full, auto_crop_length)		# returns crop in seconds
		sig = sig_full[int(crop[0] * WAV_SAMPLE_RATE) : int(crop[1] * WAV_SAMPLE_RATE)]
		if normalize_volume: sig = sig / np.max(sig)
		return crop, sig_full, sig

	def slice_sig(sig):
		start_pts = np.floor(np.linspace(0, len(sig), num_windows, endpoint=False)).astype(int)
		windows = [np.asarray(sig[pt:pt + window_size_samp]) for pt in start_pts]
		return windows

	def embed_sigs(windows, tau):
		worms = []
		for window in windows:
			embed(window, 'PRFCompare/temp_data/temp_worm.txt', False, tau, 2)
			worm = np.loadtxt('PRFCompare/temp_data/temp_worm.txt')
			worms.append(worm)
		return worms
	
	def get_filtrations(worms, filename):
		print 'building filtrations'
		filts = []
		for i, worm in enumerate(worms):
			print '\n============================================='
			print filename.split('/')[-1], 'worm #', i
			print '=============================================\n'
			filt = (Filtration(worm, filt_params))
			filts.append(filt)
			
			if PD_movie_int:
				if i % PD_movie_int == 0:
					pass
					make_movie_and_PD(filt, i, filename)
			
		return filts

	def get_PRFs(filename, crop_cmd, tau_cmd):
		sig = np.loadtxt(filename)
		crop, sig_full, sig = crop_sig(sig, crop_cmd, auto_crop_length)
		f_ideal, f_disp, tau = auto_tau(tau_cmd, sig, note_index, tau_T, None, filename)
		sigs = slice_sig(sig)
		worms = embed_sigs(sigs, tau)
		filts = get_filtrations(worms, filename)
		funcs = [filt.get_PRF(PRF_res) for filt in filts]
		return crop, sig_full, np.asarray(funcs)


	def plot_distances(d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2, out_filename):
		print 'plotting distances...\n'
		def plot_dists_pane(ax, d, mean, crop):
			t = np.linspace(crop[0], crop[1], num_windows, endpoint=False)
			ticks = np.linspace(crop[0], crop[1], num_windows + 1, endpoint=True)
			offset = (t[1] - t[0]) / 2
			ax.plot(t + offset, d, marker='o', linestyle='None', ms=10, zorder=3)
			ax.axhline(y=mean, linestyle='--', color='forestgreen', lw=2)
			ax.grid(axis='x')
			ax.set_xticks(ticks)
			# ax.set_xlim(left=crop[0], right=crop[1])

		fig = plt.figure(figsize=(18, 9), tight_layout=True)

		mean_1 = np.mean(d_1_vs_1)
		mean_2 = np.mean(d_2_vs_1)
		mean_3 = np.mean(d_1_vs_2)
		mean_4 = np.mean(d_2_vs_2)


		ax1 = fig.add_subplot(421)
		plot_dists_pane(ax1, d_1_vs_1, mean_1, crop_1)
		plt.setp(ax1.get_xticklabels(), visible=False)
		plt.setp(ax1.get_xticklines(), visible=False)
		ax1.set_ylim(bottom=0)

		ax2 = fig.add_subplot(422, sharey=ax1)
		plot_dists_pane(ax2, d_2_vs_1, mean_2, crop_2)
		plt.setp(ax2.get_yticklabels(), visible=False)
		# plt.setp(ax2.get_yticklines(), visible=False)
		plt.setp(ax2.get_xticklabels(), visible=False)
		plt.setp(ax2.get_xticklines(), visible=False)

		ax3 = fig.add_subplot(423, sharey=ax1, sharex=ax1)
		plot_dists_pane(ax3, d_1_vs_2, mean_3, crop_1)
		plt.setp(ax3.get_xticklabels(), visible=False)
		plt.setp(ax3.get_xticklines(), visible=False)


		ax4 = fig.add_subplot(424, sharey=ax1, sharex=ax2)
		plot_dists_pane(ax4, d_2_vs_2, mean_4, crop_2)
		plt.setp(ax4.get_yticklabels(), visible=False)
		# plt.setp(ax4.get_yticklines(), visible=False)

		plt.setp(ax4.get_xticklabels(), visible=False)
		plt.setp(ax4.get_xticklines(), visible=False)

		ax5 = fig.add_subplot(425, sharex=ax1)
		plot_waveform_zoom(ax5, sig_1_full, crop_1)
		ax5.grid(axis='x', zorder=0)


		ax6 = fig.add_subplot(426, sharex=ax2)
		plot_waveform_zoom(ax6, sig_2_full, crop_2)
		ax6.grid(axis='x', zorder=0)
		plt.setp(ax6.get_yticklabels(), visible=False)
		# plt.setp(ax6.get_yticklines(), visible=False)

		ylim = np.max(np.abs(np.append(ax5.get_ylim(), ax6.get_ylim())))
		ax5.set_ylim(-ylim, ylim)
		ax6.set_ylim(-ylim, ylim)


		ax7 = fig.add_subplot(427)
		plot_waveform(ax7, sig_1_full, crop_1)


		ax8 = fig.add_subplot(428, sharey=ax7)
		plot_waveform(ax8, sig_2_full, crop_2)
		plt.setp(ax8.get_yticklabels(), visible=False)
		plt.setp(ax8.get_yticklines(), visible=False)



		ax1.set_title(filename_1.split('/')[-1])
		ax2.set_title(filename_2.split('/')[-1])

		del_12 = mean_2 - mean_1
		del_34 = mean_4 - mean_3

		ax1.set_ylabel('\n \n ref: ' + filename_1.split('/')[-1].split('.')[0] + ' \n \n $\Delta$: {:.3f}'.format(del_12), rotation=0, size='large', labelpad=50)
		ax3.set_ylabel('\n \n ref: ' + filename_2.split('/')[-1].split('.')[0] + ' \n \n $\Delta$: {:.3f}'.format(del_34), rotation=0, size='large', labelpad=50)


		plt.savefig(out_filename)


		plt.close(fig)

	# ===========================================================================
	# 		mean_PRF_dist_plots()
	# ===========================================================================

	clear_old_files('output/PRFCompare/mean_PRFC/PDs_and_movies/', PD_movie_int)
	filt_params.update({'worm_length' : np.floor(window_size * WAV_SAMPLE_RATE).astype(int)})
	print 'using worm_length:', filt_params['worm_length']
	window_size_samp = int(window_size * WAV_SAMPLE_RATE)
	crop_1_cmd, crop_2_cmd = crop_1, crop_2
	tau_1_cmd, tau_2_cmd = tau_1, tau_2



	if load_saved_filtrations:
		crop_1, sig_1_full, funcs_1 = np.load('PRFCompare/funcs_1.npy')
		crop_2, sig_2_full, funcs_2 = np.load('PRFCompare/funcs_2.npy')
	else:
		crop_1, sig_1_full, funcs_1 = get_PRFs(filename_1, crop_1_cmd, tau_1_cmd)
		crop_2, sig_2_full, funcs_2 = get_PRFs(filename_2, crop_2_cmd, tau_2_cmd)
		np.save('PRFCompare/funcs_1.npy', (crop_1, sig_1_full, funcs_1))
		np.save('PRFCompare/funcs_2.npy', (crop_1, sig_1_full, funcs_2))


	funcs_1_z = funcs_1[:, 2]
	funcs_2_z = funcs_2[:, 2]

	mean_1_funcs_z = funcs_1_z[::int(ceil(num_windows/mean_samp_num))]
	mean_2_funcs_z = funcs_2_z[::int(ceil(num_windows/mean_samp_num))]

	funcs_1_avg_z = np.mean(mean_1_funcs_z, axis=0)
	funcs_2_avg_z = np.mean(mean_2_funcs_z, axis=0)

	dists_1_vs_1 = get_scaled_dists(funcs_1_z, funcs_1_avg_z, weight_func, metric, dist_scale, PRF_res)
	dists_2_vs_1 = get_scaled_dists(funcs_2_z, funcs_1_avg_z, weight_func, metric, dist_scale, PRF_res)
	dists_1_vs_2 = get_scaled_dists(funcs_1_z, funcs_2_avg_z, weight_func, metric, dist_scale, PRF_res)
	dists_2_vs_2 = get_scaled_dists(funcs_2_z, funcs_2_avg_z, weight_func, metric, dist_scale, PRF_res)

	plot_distances(dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2, out_filename)

	# plot ref PRFs #
	ref_func_1 = funcs_1[0]  # get xx, yy
	ref_func_2 = funcs_2[0]  # get xx, yy
	ref_func_1[2] = funcs_1_avg_z
	ref_func_2[2] = funcs_2_avg_z

	base_filename_1 = filename_1.split('/')[-1].split('.')[0]
	base_filename_2 = filename_2.split('/')[-1].split('.')[0]
	out_fname_1 =  'output/PRFCompare/mean_PRFC/MEAN_PRF_' + base_filename_1 + '.png'
	out_fname_2 = 'output/PRFCompare/mean_PRFC/MEAN_PRF_' + base_filename_2 + '.png'
	make_PRF_plot(ref_func_1, out_fname_1, params=filt_params, in_filename='MEAN')
	make_PRF_plot(ref_func_2, out_fname_2, params=filt_params, in_filename='MEAN')
	# end plot ref PRFs #



