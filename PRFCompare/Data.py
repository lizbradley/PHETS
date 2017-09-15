from __future__ import division

import sys
import warnings
from math import ceil

import numpy as np

from DCE.DCE import embed
from DCE.Tools import auto_crop
from DCE.Tools import auto_tau
from PH.Data import Filtration
from PH.FiltrationMovie import make_movie
from PH.Plots import make_PD, make_PRF_plot
from Utilities import clear_old_files, blockPrint, enablePrint, print_title
from config import WAV_SAMPLE_RATE

def apply_weight_func(f, weight_func):

	# x, y, max_lim included vs z only
	f_format_full = len(f.shape) == 1 and f.size == 4

	z = f[2] if f_format_full else f

	x = y = np.linspace(0, 2 ** .5, len(z))
	xx, yy = np.meshgrid(x, y)

	weight_func = weight_func(xx, yy)
	if isinstance(weight_func, int):
		weight_func = xx * 0 + weight_func

	z = np.multiply(z, weight_func)

	if f_format_full:
		f[2] = z
		return f
	else:
		return z



def norm(f, metric):

	prf_res = len(f)
	dA = 2. / (prf_res ** 2)	# normalize such that area of PRF domain is 1

	if metric == 'L1':
		return np.nansum(np.abs(f)) * dA

	elif metric == 'L2':
		return np.sqrt(np.nansum(np.power(f, 2)) * dA)

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
		print "ERROR: dist_scale '" + scale + \
			  "' is not recognized. Use 'none', 'a', 'b', or 'a + b'."
		sys.exit()



def get_dists_from_ref(funcs, ref_func, metric, scale):
	dists = [norm(np.subtract(f, ref_func), metric) for f in funcs]
	norms = [norm(f, metric) for f in funcs]
	norm_ref = [norm(ref_func, metric)] * len(dists)
	return scale_dists(dists, norms, norm_ref, scale)


def get_PRFs(
		filename, filt_params, crop_cmd, tau_cmd,
		PRF_res,
		auto_crop_length,
		time_units,
		normalize_volume,
		normalize_win_vol,
		mean_samp_num,
		num_windows,
		window_size,
		see_samples,
		note_index,
		tau_T,
		fname=None
		):

	"""helper for dists_compare"""

	def show_samples(filt, i, filename):
		base_name = filename.split('/')[-1].split('.')[0]
		comp_name = 'output/PRFCompare/mean/see_samples/{:s}_{:d}_'.format(
			base_name, i
		)
		PD_filename = comp_name + 'PD.png'
		PRF_filename = comp_name + 'PRF.png'
		movie_filename = comp_name + 'movie.mp4'

		make_PD(filt, PD_filename)
		make_PRF_plot(filt, PRF_filename)
		make_movie(filt, movie_filename)


	def crop_sig(sig_full, crop_cmd, auto_crop_len):
		crop = auto_crop(crop_cmd, sig_full, auto_crop_length, time_units)
		crop_samp = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		sig = sig_full[crop_samp[0]:crop_samp[1]]
		if normalize_volume: sig = sig / np.max(sig)
		return crop, sig_full, sig


	def slice_sig(sig):
		if mean_samp_num > num_windows:
			print 'ERROR: mean_samp_num may not exceed num_windows'
			sys.exit()

		start_pts = np.floor(
			np.linspace(0, len(sig) - 1, num_windows, endpoint=False)
		).astype(int)

		windows = [np.asarray(sig[pt:pt + window_size_samp])
				   for pt in start_pts]

		if normalize_win_vol:
			windows = [np.true_divide(w, np.max(np.abs(w))) for w in windows]

		return windows


	def embed_sigs(windows, tau):
		worms = []
		for window in windows:
			worm = embed(window, tau, 2, time_units='seconds')
			worms.append(worm)
		return worms


	def get_filtrations(worms, filename):
		print 'building filtrations'
		filts = []
		for i, worm in enumerate(worms):
			print '\n============================================='
			print filename.split('/')[-1], 'worm #', i
			print '=============================================\n'

			# blockPrint()
			filt = (Filtration(worm, filt_params, title=filename))
			# enablePrint()

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
			funcs.append(filt.get_PRF())
		return np.asarray(funcs)




	# =========================================================================
	# 		MAIN: get_PRFs()
	# =========================================================================

	if window_size:
		if time_units == 'samples':
			window_size_samp = window_size
		else:
			window_size_samp = int(window_size * WAV_SAMPLE_RATE)
		filt_params.update({'worm_length': window_size_samp})

	else:
		window_size_samp = filt_params['worm_length']

	print 'using window_size_samp:', window_size_samp

	if isinstance(filename, basestring):
		print 'loading', filename, '...'
		sig = np.loadtxt(filename)
	else:
		sig = filename
		filename=fname

	crop, sig_full, sig = crop_sig(sig, crop_cmd, auto_crop_length)
	f_ideal, f_disp, tau = auto_tau(tau_cmd, sig, note_index, tau_T, None,
									filename)


	sigs = slice_sig(sig)
	if type(sig[0]) is np.ndarray:
		dim = len(sig[0])
		print 'Input dimension: ' + str(dim) + '. Skipping embedding, dropping\
		 coords for time series.'
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
		normalize_win_volume=True,

		PRF_res=50,  # number of divisions used for PRF
		dist_scale='none',  # 'none', 'a', or 'a + b'
		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		weight_func=lambda i, j: 1,

		see_samples=5

):
	'''generates and processes data for plot_dists_vs_ref, plot_dists_vs_means,
	 and plot_clusters'''

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
			normalize_win_volume,
			mean_samp_num,
			num_windows,
			window_size,
			see_samples,
			note_index,
			tau_T,
		]
		crop_1, sig_1_full, sig_1, funcs_1 = get_PRFs(
			filename_1, filt_params, crop_1_cmd, tau_1_cmd, *options
		)
		crop_2, sig_2_full, sig_2, funcs_2 = get_PRFs(
			filename_2, filt_params, crop_2_cmd, tau_2_cmd, *options
		)
		np.save('PRFCompare/funcs_1.npy', (crop_1, sig_1_full, sig_1, funcs_1))
		np.save('PRFCompare/funcs_2.npy', (crop_2, sig_2_full, sig_2, funcs_2))

	funcs_1_z = funcs_1[:, 2]
	funcs_2_z = funcs_2[:, 2]

	mean_1_funcs_z = funcs_1_z[::int(ceil(num_windows / mean_samp_num))]
	mean_2_funcs_z = funcs_2_z[::int(ceil(num_windows / mean_samp_num))]


	funcs_1_avg_z = np.mean(mean_1_funcs_z, axis=0)
	funcs_2_avg_z = np.mean(mean_2_funcs_z, axis=0)



	np.savetxt('PRFCompare/text_data/mean_prf_1.txt', funcs_1_avg_z)
	np.savetxt('PRFCompare/text_data/mean_2.txt', funcs_2_avg_z)

	d_s = dist_scale
	dists_1_vs_1 = get_dists_from_ref(funcs_1_z, funcs_1_avg_z, metric, d_s)
	dists_2_vs_1 = get_dists_from_ref(funcs_2_z, funcs_1_avg_z, metric, d_s)
	dists_1_vs_2 = get_dists_from_ref(funcs_1_z, funcs_2_avg_z, metric, d_s)
	dists_2_vs_2 = get_dists_from_ref(funcs_2_z, funcs_2_avg_z, metric, d_s)


	np.savetxt('PRFCompare/text_data/dist_1_vs_1.txt', dists_1_vs_1)
	np.savetxt('PRFCompare/text_data/dist_2_vs_1.txt', dists_2_vs_1)
	np.savetxt('PRFCompare/text_data/dist_1_vs_2.txt', dists_1_vs_2)
	np.savetxt('PRFCompare/text_data/dist_2_vs_2.txt', dists_2_vs_2)


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





###################################################################




def get_prf_evo(sig, filt_params, num_windows, silent=True):

	def slice_sig(sig, num_windows, window_len_samp):
		start_pts = np.floor(
			np.linspace(0, len(sig), num_windows, endpoint=False)
		).astype(int)
		windows = [np.asarray(sig[pt : pt + filt_params['worm_length']])
				   for pt in start_pts]
		# TODO: add normalize sub volume here
		return windows


	def get_filtrations(worms, filt_params):
		filts = []
		for i, worm in enumerate(worms):
			if silent:
				blockPrint()
				filt = Filtration(worm, filt_params, silent=silent)
				enablePrint()
			else:
				print_title('window # {}'.format(i))
				filt = Filtration(worm, filt_params, silent=silent)

			filts.append(filt)

		return filts


	def get_prfs(filts):
		funcs = []
		for i, filt in enumerate(filts):
			if silent:
				funcs.append(filt.get_PRF(silent=True))
			else:
				print_title('window # {}'.format(i))
				funcs.append(filt.get_PRF(silent=False))

		return np.asarray(funcs)





	sigs = slice_sig(sig, num_windows, filt_params['worm_length'])
	filts = get_filtrations(sigs, filt_params)
	prfs = get_prfs(filts)
	return prfs, filts



def crop_sig(sig, crop, time_units):

	if crop is None:
		return sig
	elif time_units == 'samples':
		return sig[crop[0]: crop[1]]
	elif time_units == 'seconds':
		crop_samp = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		return sig[crop_samp[0]:crop_samp[1]]
	else:
		print "ERROR: invalid 'crop' or 'time_units'"
		sys.exit()


def get_variance_data(filename, kwargs):


	if kwargs['load_saved_filts']:
		print 'WARNING: loading saved data'
		return np.load('PRFCompare/PRFs.npy'), np.load('PRFCompare/filts.npy')


	print 'loading', filename, '...'
	sig_full = np.loadtxt(filename)

	vary_param_1 = kwargs['vary_param_1']
	vary_param_2 = kwargs['vary_param_2']
	filt_params = kwargs['filt_params']


	if len(sig_full.shape) == 1:
		print 'Input has shape {}. Embedding...'
		# sig_full = embed(sig_full, None, kwargs['m'], kwargs['tau'])
		sig_full = embed(
			sig_full, kwargs['tau'], kwargs['m'], kwargs['time_units']
		)

	sig = crop_sig(sig_full, kwargs['crop'], kwargs['time_units'])

	def vary_evos_over_param(sig, vary_param, filt_params, title_str=None):
		prf_arr = []
		filt_arr = []
		for val_1 in vary_param[1]:
			filt_params.update({vary_param[0]: val_1})

			if kwargs['quiet']:
				sys.stdout.write(
					'\r		vary_param_1: {}		vary_param_2: {}'.format(
						val_1, val_2
					)
				)
				sys.stdout.flush()
				# get PRFs at evenly spaced intervals along input
				# ie a 'prf evolution'
				prf_evo, filt_evo = get_prf_evo(
					sig, filt_params, kwargs['num_windows'], silent=True
				)

			else:
				print_title('{}: {}'.format(vary_param_1[0], val_1))
				if title_str: print_title(title_str)
				prf_evo, filt_evo = get_prf_evo(
					sig, filt_params, kwargs['num_windows'], silent=False
				)

			filt_arr.append(filt_evo)
			prf_arr.append(prf_evo)

		return prf_arr, filt_arr


	print 'generating data...\n'

	if vary_param_2 is None or vary_param_2[0] == 'weight_func':
		val_2 = ''		# for status indicator
		prf_evos, filt_evos = vary_evos_over_param(
			sig, vary_param_1, filt_params
		)
		sys.stdout.write('\r	done \n\n')
		sys.stdout.flush()
		prf_evos, filt_evos = [prf_evos], [filt_evos]

	elif vary_param_2[0] in filt_params:
		prf_arr = []
		filt_arr = []
		for val_2 in vary_param_2[1]:
			filt_params.update({vary_param_2[0]: val_2})
			t_str = '{}: {}'.format(vary_param_2[0], val_2)
			prf_evos, filt_evos = vary_evos_over_param(
				sig, vary_param_1, filt_params, title_str=t_str
			)

			prf_arr.append(prf_evos)
			filt_arr.append(filt_evos)

		prf_evos = prf_arr
		filt_evos = filt_arr
		sys.stdout.write('\r	done \n\n')
		sys.stdout.flush()

	else:
		print 'ERROR: invalid vary_param_2 '
		sys.exit()

	prf_evos = np.asarray(prf_evos)
	filt_evos = np.asarray(filt_evos)
	np.save('PRFCompare/PRFs.npy', prf_evos)
	np.save('PRFCompare/filts.npy', filt_evos)
	return prf_evos, filt_evos




class VarianceData:
	"""all data for a fixed value of vary_param_2 -- one curve per plot"""
	def __init__(self):
		self.pointwise_mean_norm = []
		self.variance = []
		self.scaled_variance = []
		self.pointwise_variance_norm = []
		self.functional_COV_norm = []

class HeatmapData:

	def __init__(self):
		self.pointwise_mean = [[]]
		self.pointwise_var = [[]]
		self.functional_COV = [[]]



def process_variance_data(prf_evo_array, metric, weight_func, vary_param_2):

	def apply_weight_to_evo(prf_evo, weight_f):
		weighted_prf_evo = []
		for prf in prf_evo:
			weighted_prf_evo.append(apply_weight_func(prf, weight_f))
		return np.asarray(weighted_prf_evo)


	def apply_weight_func_to_array(prf_evo_array, weight_f):
		for row in prf_evo_array:
			for evo in row:
				evo[...] = apply_weight_to_evo(evo, weight_f)
		return prf_evo_array


	def vary_evos_over_weight_func(prf_evos):
		prf_evos_1d = prf_evos[0]
		prf_evos_2d = []
		for prf_evo in prf_evos_1d:
			prf_evo_vary_2 = []
			for weight_f in vary_param_2[1]:
				weighted_prf_evo = apply_weight_to_evo(prf_evo, weight_f)
				prf_evo_vary_2.append(weighted_prf_evo)
			prf_evos_2d.append(prf_evo_vary_2)

		prf_evos_2d = np.transpose(np.asarray(prf_evos_2d), (1, 0, 2, 3))
		return prf_evos_2d


	def calculate_stats(prf_evos_1d, apply_weight_to_fcov=True):
		var_data = VarianceData()
		hmap_data_arr = []

		for prf_evo in prf_evos_1d:  # for each value of vary_param_1

			hmap_data = HeatmapData()
			# see definitions for norm() and get_dists_from_ref() around lines 45 - 90

			prf_evo = prf_evo[:, 2]  # take z component only
			pointwise_mean = np.mean(prf_evo, axis=0)  				# plot as heatmap
			hmap_data.pointwise_mean = pointwise_mean

			pmn = norm(pointwise_mean, metric)  					# plot as data point
			var_data.pointwise_mean_norm.append(pmn)

			dists = [norm(np.subtract(PRF, pointwise_mean), metric)
					 for PRF in prf_evo]
			variance = np.mean(np.power(dists, 2))  				# plot as data point
			# variance = np.sum(np.power(dists, 2)) / (len(dists) - 1)
			var_data.variance.append(variance)

			scaled_dists = get_dists_from_ref(
				prf_evo, pointwise_mean, metric, 'b'
			)
			scaled_variance = np.mean(np.power(scaled_dists, 2))    # plot as data point
			var_data.scaled_variance.append(scaled_variance)

			diffs = [PRF - pointwise_mean for PRF in prf_evo]

			pointwise_variance = np.var(diffs, axis=0) 				# plot as heatmap
			hmap_data.pointwise_var = pointwise_variance

			pvn = norm(pointwise_variance, metric) 					# plot as data point
			var_data.pointwise_variance_norm.append(pvn)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				FCOV = pointwise_variance / pointwise_mean  		 # plot as heatmap

			if apply_weight_to_fcov:
				hmap_data.functional_COV = apply_weight_func(FCOV, weight_func)
			else:
				hmap_data.functional_COV = FCOV

			fcovn = norm(FCOV, metric)  # plot as data point
			var_data.functional_COV_norm.append(fcovn)

			hmap_data_arr.append(hmap_data)

		return var_data, hmap_data_arr



	print 'processing data...'

	prf_evo_array_pre_weight = prf_evo_array

	if vary_param_2 and vary_param_2[0] == 'weight_func':
		prf_evo_array = vary_evos_over_weight_func(prf_evo_array)
	elif vary_param_2:
		prf_evo_array = apply_weight_func_to_array(prf_evo_array, weight_func)


	curve_data = []
	hmap_data = []
	for row in prf_evo_array:
		cd, hmd = calculate_stats(row)
		curve_data.append(cd)
		hmap_data.append(hmd)

	hmap_data_pre_weight = []
	for row in prf_evo_array_pre_weight:
		cd, hmd_pre_weight = calculate_stats(row, apply_weight_to_fcov=False)
		hmap_data_pre_weight.append(hmd_pre_weight)

	curve_data = np.asarray(curve_data)
	hmap_data = np.asarray(hmap_data)
	hmap_data_pre_weight = np.asarray(hmap_data_pre_weight)

	return curve_data, hmap_data, hmap_data_pre_weight




