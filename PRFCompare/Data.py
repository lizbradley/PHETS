from __future__ import division

import sys
from math import ceil

import numpy as np

from DCE.DCE import embed
from DCE.Tools import auto_crop
from DCE.Tools import auto_tau
from PH.Data import Filtration
from PH.FiltrationMovie import make_movie
from PH.Plots import make_PD, make_PRF_plot
from Utilities import clear_old_files, blockPrint, enablePrint

from config import WAV_SAMPLE_RATE

def norm(f, metric, f_weight):

	PRF_res = len(f)
	dA = 2. / (PRF_res ** 2)		# normalize such that area of PRF domain is 1
	f_weight = f_weight(*np.meshgrid(np.linspace(0, 1, PRF_res), np.linspace(0, 1, PRF_res)))

	if metric == 'L1':
		return np.nansum(np.multiply(np.abs(f), f_weight)) * dA

	elif metric == 'L2':
		return np.sqrt(np.nansum(np.multiply(np.power(f, 2), f_weight))) * dA

	else:
		print "ERROR: metric not recognized. Use 'L1' or 'L2'."
		sys.exit()



def scale_dists(dists, norms, norm_ref, scale):
	""" helper for get_dists_from_ref """
	if scale == 'none':
		return dists

	elif scale == 'a':
		return np.true_divide(dists, norms)

	elif scale == 'b':
		return np.true_divide(dists, norm_ref)

	elif scale == 'a + b':
		return np.true_divide(dists, np.add(norms, norm_ref))

	else:
		print "ERROR: dist_scale '" + scale + "' is not recognized. Use 'none', 'a', 'b', or 'a + b'."
		sys.exit()



def get_dists_from_ref(funcs, ref_func, weight_func, metric, scale):
	dists = [norm(np.subtract(f, ref_func), metric, weight_func) for f in funcs]
	norms = [norm(f, metric, lambda i, j: 1) for f in funcs]
	norm_ref = [norm(ref_func, metric, lambda i, j: 1)] * len(dists)
	return scale_dists(dists, norms, norm_ref, scale)



def get_PRFs(
		filename, filt_params, crop_cmd, tau_cmd,
		PRF_res,
		auto_crop_length,
		time_units,
		normalize_volume,
		mean_samp_num,
		num_windows,
		window_size,
		see_samples,
		note_index,
		tau_T,
		fname=None
		):

	def show_samples(filt, i, filename):
		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = '{:s}_{:d}_'.format(base_name, i)
		PD_filename = 'output/PRFCompare/mean/see_samples/' + comp_name + 'PD.png'
		PRF_filename = 'output/PRFCompare/mean/see_samples/' + comp_name + 'PRF.png'
		movie_filename = 'output/PRFCompare/mean/see_samples/' + comp_name + 'movie.mp4'

		make_PD(filt, PD_filename)
		make_PRF_plot(filt, PRF_filename, PRF_res=PRF_res)
		make_movie(filt, movie_filename)


	def crop_sig(sig_full, crop_cmd, auto_crop_len):

		crop = auto_crop(crop_cmd, sig_full, auto_crop_length, time_units=time_units)  # returns crop in seconds

		sig = sig_full[int(crop[0] * WAV_SAMPLE_RATE): int(crop[1] * WAV_SAMPLE_RATE)]

		if normalize_volume: sig = sig / np.max(sig)
		return crop, sig_full, sig


	def slice_sig(sig):
		if mean_samp_num > num_windows:
			print 'ERROR: mean_samp_num may not exceed num_windows'
			sys.exit()
		start_pts = np.floor(np.linspace(0, len(sig) - 1, num_windows, endpoint=False)).astype(int)
		windows = [np.asarray(sig[pt:pt + window_size_samp]) for pt in start_pts]
		# TODO: add normalize sub volume here
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

			blockPrint()
			filt = (Filtration(worm, filt_params, filename=filename))
			enablePrint()

			filts.append(filt)

			if see_samples:
				if i % see_samples == 0:
					show_samples(filt, i, filename)

		return filts


	def get_funcs(filts, filename):
		funcs = []
		for i, filt in enumerate(filts):
			print '\n============================================='
			print filename.split('/')[-1], 'worm #', i
			print '=============================================\n'
			funcs.append(filt.get_PRF(PRF_res))
		return np.asarray(funcs)




	# ===========================================================================
	# 		MAIN: get_PRFs()
	# ===========================================================================

	if window_size:
		window_size_samp = window_size if time_units == 'samples' else int(window_size * WAV_SAMPLE_RATE)
		filt_params.update({'worm_length': window_size_samp})
	else:
		window_size_samp = filt_params['worm_length']

	if isinstance(filename, basestring):
		print 'loading', filename, '...'
		sig = np.loadtxt(filename)
	else:
		sig = filename
		filename=fname

	crop, sig_full, sig = crop_sig(sig, crop_cmd, auto_crop_length)
	f_ideal, f_disp, tau = auto_tau(tau_cmd, sig, note_index, tau_T, None, filename)


	sigs = slice_sig(sig)
	if type(sig[0]) is np.ndarray:
		dim = len(sig[0])
		print 'Input dimension: ' + str(dim) + '. Skipping embedding, dropping coords for time series.'
		worms = sigs
		sig_full = sig_full[:, 0]
	elif type(sig[0] is np.float64):
		dim = 1
		print 'Input dimension: ' + str(dim) + '. Embedding signals...'
		worms = embed_sigs(sigs, tau)

	filts = get_filtrations(worms, filename)
	funcs = get_funcs(filts, filename)

	return crop, sig_full, sig, np.asarray(funcs)



def dists_compare(
		filename_1,
		filename_2,
		out_filename,
		filt_params,

		load_saved_PRFs=False,

		time_units='seconds',

		crop_1='auto',
		crop_2='auto',
		auto_crop_length=.3,

		window_size=None,
		num_windows=10,  # per file
		mean_samp_num=5,  # per file

		tau_1=.001,
		tau_2=.001,
		tau_T=np.pi,
		note_index=None,  #

		normalize_volume=True,

		PRF_res=50,  # number of divisions used for PRF
		dist_scale='none',  # 'none', 'a', or 'a + b'
		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		weight_func=lambda i, j: 1,

		see_samples=5

):

	clear_old_files('output/PRFCompare/mean/see_samples/', see_samples)

	crop_1_cmd, crop_2_cmd = crop_1, crop_2
	tau_1_cmd, tau_2_cmd = tau_1, tau_2

	if load_saved_PRFs:
		print 'WARNING: loading saved filtration'
		crop_1, sig_1_full, sig_1, funcs_1 = np.load('PRFCompare/funcs_1.npy')
		crop_2, sig_2_full, sig_2, funcs_2 = np.load('PRFCompare/funcs_2.npy')
	else:
		options = [
			PRF_res,
			auto_crop_length,
			time_units,
			normalize_volume,
			mean_samp_num,
			num_windows,
			window_size,
			see_samples,
			note_index,
			tau_T
		]
		crop_1, sig_1_full, sig_1, funcs_1 = get_PRFs(filename_1, filt_params, crop_1_cmd, tau_1_cmd, *options)
		crop_2, sig_2_full, sig_2, funcs_2 = get_PRFs(filename_2, filt_params, crop_2_cmd, tau_2_cmd, *options)
		np.save('PRFCompare/funcs_1.npy', (crop_1, sig_1_full, sig_1, funcs_1))
		np.save('PRFCompare/funcs_2.npy', (crop_2, sig_2_full, sig_2, funcs_2))

	funcs_1_z = funcs_1[:, 2]
	funcs_2_z = funcs_2[:, 2]

	mean_1_funcs_z = funcs_1_z[::int(ceil(num_windows / mean_samp_num))]
	mean_2_funcs_z = funcs_2_z[::int(ceil(num_windows / mean_samp_num))]

	funcs_1_avg_z = np.mean(mean_1_funcs_z, axis=0)
	funcs_2_avg_z = np.mean(mean_2_funcs_z, axis=0)

	dists_1_vs_1 = get_dists_from_ref(funcs_1_z, funcs_1_avg_z, weight_func, metric, dist_scale)
	dists_2_vs_1 = get_dists_from_ref(funcs_2_z, funcs_1_avg_z, weight_func, metric, dist_scale)
	dists_1_vs_2 = get_dists_from_ref(funcs_1_z, funcs_2_avg_z, weight_func, metric, dist_scale)
	dists_2_vs_2 = get_dists_from_ref(funcs_2_z, funcs_2_avg_z, weight_func, metric, dist_scale)


	# for plotting ref PRFs #
	ref_func_1 = funcs_1[0]  # get xx, yy
	ref_func_2 = funcs_2[0]  # get xx, yy
	ref_func_1[2] = funcs_1_avg_z
	ref_func_2[2] = funcs_2_avg_z


	arr = [
		[sig_1_full, sig_2_full],
		[crop_1, crop_2],
		[sig_1, sig_2],
		[ref_func_1, ref_func_2],
		[dists_1_vs_1, dists_2_vs_1, dists_1_vs_2, dists_2_vs_2]
	]

	return arr


