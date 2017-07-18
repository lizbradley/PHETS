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


	"""helper for dists_compare"""
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
	'''generates and processes data for plot_dists_vs_ref, plot_dists_vs_means, and plot_clusters'''

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





###################################################################


class VarianceData:  	# all data for a fixed value of vary_param_2 -- one curve per plot
	def __init__(self):
		self.mean_PRF_norm = []
		self.variance = []
		self.scaled_variance = []
		self.pointwise_variance_norm = []
		self.functional_COV_norm = []

def plot_trajectory(sig):
	print 'plotting trajectory...'
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111)
	cbar = ax.scatter(sig[:,0], sig[:,1], s=.05, c=np.arange(sig.shape[0]))
	fig.colorbar(cbar)
	fig.savefig('output/PRFCompare/variance/trajectory.png')


def get_variance_data(filename, vary_param_1, vary_param_2, filt_params, crop, tau, load_saved_PRFs=False):

	if load_saved_PRFs:
		print 'WARNING: loading saved data'
		return np.load('PRFCompare/PRFs.npy')

	if vary_param_2:
		if vary_param_2[0] in filt_params:
			print 'loading', filename, '...'
			sig_full = np.loadtxt(filename)
			arr_2 = []
			for val_2 in vary_param_2[1]:
				filt_params.update({vary_param_2[0]: val_2})
				arr_1 = []
				for val_1 in vary_param_1[1]:

					filt_params.update({vary_param_1[0]: val_1})

					# get PRFs at evenly spaced intervals along input -- 'prf evolution'

					sys.stdout.flush()
					sys.stdout.write('\rComputing data... vary_param 1: {}, vary_param_2: {}'.format(val_1, val_2))
					sys.stdout.flush()


					# plot_trajectory(sig)
					# arr_1.append(prf_evo[:, 2])		# take z component only
					arr_1.append(prf_evo)
				arr_2.append(arr_1)

			prf_evos = np.asarray(arr_2)
			# array with shape (len(vary_param_2[1], len(vary_param_1[1])
			# each element is a 1D list of PRFs from regular samples of the input
			np.save('PRFCompare/PRFs.npy', prf_evos)
			return prf_evos

		else:
			print 'ERROR: currently vary_param_2 must be a filtration parameter'
			sys.exit()

def process_data(prf_evo_array):
	print 'processing...'
	data_list = []
	for i, row in enumerate(prf_evo_array):  # for each value of vary_param_2

		data_val_2 = VarianceData()

		for j, sample_prfs in enumerate(row):	 # for each value of vary_param_1

			# see definitions for norm() and get_dists_from_ref() around lines 45 - 90

			sample_prfs_z = sample_prfs[:, 2]		# take z component only
			null_weight_func = lambda i, j: 1
			pointwise_mean = np.mean(sample_prfs_z, axis=0)					# plot as heatmap
			pmn = norm(pointwise_mean, metric, null_weight_func) 			# plot as data point
			data_val_2.mean_PRF_norm.append(pmn)

			# HOMEGROWN VARIANCE #

			dists = [norm(np.subtract(PRF, pointwise_mean), metric, weight_func) for PRF in sample_prfs_z]
			variance = np.mean(dists)										# plot as data point
			data_val_2.variance.append(variance)

			scaled_dists = get_dists_from_ref(sample_prfs_z, pointwise_mean, weight_func, metric, dist_scale)
			scaled_variance = np.mean(scaled_dists)							# plot as data point
			data_val_2.scaled_variance.append(scaled_variance)

			# POINTWISE VARIANCE #

			diffs = [PRF - pointwise_mean for PRF in sample_prfs_z]

			pointwise_variance = np.var(diffs, axis=0)						# plot as heatmap
			pvn = norm(pointwise_variance, metric, null_weight_func)		# plot as data point
			data_val_2.pointwise_variance_norm.append(pvn)

			functional_COV = pointwise_variance / pointwise_mean			# plot as heatmap
			fcovn = norm(functional_COV, metric, null_weight_func)			# plot as data point
			data_val_2.functional_COV_norm.append(fcovn)



			# x, y, z, max_lim = sample_prfs[0]
			#
			# fig = plt.figure(figsize=(13, 5), tight_layout=True)
			# ax1 = 		fig.add_subplot(131)
			# ax2 = 		fig.add_subplot(132)
			# ax3 = 		fig.add_subplot(133)
			# divider = make_axes_locatable(ax3)
			# c_ax = divider.append_axes('right', size='10%', pad=.2)
			#
			# plot_heatmap(ax1, c_ax, x, y, pointwise_mean)
			# ax1.set_title('pointwise mean')
			# plot_heatmap(ax2, c_ax, x, y, pointwise_variance)
			# ax2.set_title('pointwise variance')
			# plot_heatmap(ax3, c_ax, x, y, functional_COV)
			# ax3.set_title('functional COV')
			#
			# fig.suptitle(filename.split('/')[-1])
			# fname = '{}_{}__{}_{}.png'.format(vary_param_2[0], vary_param_2[1][i], vary_param_1[0], vary_param_1[1][j])
			# fig.savefig('output/PRFCompare/variance/heatmaps/' + fname)
			# plt.close(fig)



		data_list.append(data_val_2)

	return data_list
