import sys, os, shutil, inspect


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from DCE.Plotter import plot_waveform_zoom, plot_waveform
from PH import make_PD, make_PRF_plot, make_movie, Filtration
from PH.Plots import plot_heatmap
from PH.TitleBox import add_filenames_table, add_filt_params_table
from PRFCompare.Data import get_dists_from_ref, dists_compare
from Utilities import clear_old_files, clear_dir, print_title, lambda_to_str


def plot_dists_vs_ref(
		dir, base_filename,
		fname_format,
		out_filename,
		filt_params,

		i_ref=15,
		i_arr=np.arange(10, 20, 1),

		weight_func=lambda i, j: 1,

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', or 'a + b'

		PRF_res=50,  # number of divisions used for PRF

		load_saved_PRFs=False,

		see_samples=5,
):
	""" plots distance from reference rank function over a range of trajectories input files"""

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

	def show_samples(filt, i, ref=False):
		base_name = base_filename.split('/')[-1].split('.')[0]
		comp_name = '{:s}_{:d}_'.format(base_name, i)
		if ref: comp_name += '_REFERENCE_'
		PD_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'movie.mp4'
		PRF_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'PRF.png'

		make_PD(filt, PD_filename)
		make_PRF_plot(filt, PRF_filename)
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

			if see_samples:
				if i % see_samples == 0:
					show_samples(filt, i)

		return np.asarray(funcs)

	# ===================================================== #
	# 				MAIN:	plot_dists_vs_ref()				#
	# ===================================================== #

	clear_old_files('output/PRFCompare/ref/see_samples/', see_samples)

	if load_saved_PRFs:
		print 'WARNING: loading saved filtration'
		funcs = np.load('PRFCompare/funcs.npy')
		ref_func = np.load('PRFCompare/ref_func.npy')
	else:
		funcs = get_PRFs()  # also makes PDs and movies
		ref_filt = Filtration(get_in_filename(i_ref), filt_params)
		if see_samples: show_samples(ref_filt, i_ref, ref=True)
		ref_func = ref_filt.get_PRF(PRF_res)
		np.save('PRFCompare/funcs.npy', funcs)
		np.save('PRFCompare/ref_func.npy', ref_func)

	make_PRF_plot(
		ref_func,
		'output/PRFCompare/ref/PRF_REFERENCE.png',
		params=filt_params,
		in_filename='REF'
	)

	funcs_z = funcs[:, 2]
	ref_func_z = ref_func[2]
	dists = get_dists_from_ref(funcs_z, ref_func_z, weight_func, metric, dist_scale)
	plot_distances(i_ref, i_arr, dists, out_filename)



def plot_dists_vs_means(*args, **kwargs):		# see dists_compare for arg format

	def plot():
		def plot_dists_pane(ax, d, mean, crop):
			t = np.linspace(crop[0], crop[1], num_windows, endpoint=False)
			ticks = np.linspace(crop[0], crop[1], num_windows + 1, endpoint=True)
			offset = (t[1] - t[0]) / 2
			ax.plot(t + offset, d, marker='o', linestyle='None', ms=10, zorder=3)
			ax.axhline(y=mean, linestyle='--', color='forestgreen', lw=2)
			ax.grid(axis='x')
			ax.set_xticks(ticks)

		print 'plotting distances...'

		d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists
		crop_1, crop_2 = crops
		sig_1_full, sig_2_full = sigs_full
		sig_1, sig_2 = sigs

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
		plot_waveform_zoom(ax5, None, crop_1, time_units=time_units, sig=sig_1)
		ax5.grid(axis='x', zorder=0)
		plt.setp(ax5.get_yticklabels(), visible=False)
		plt.setp(ax5.get_yticklines(), visible=False)



		ax6 = fig.add_subplot(426, sharex=ax2)
		plot_waveform_zoom(ax6, None, crop_2, time_units=time_units, sig=sig_2)
		ax6.grid(axis='x', zorder=0)
		plt.setp(ax6.get_yticklabels(), visible=False)
		plt.setp(ax6.get_yticklines(), visible=False)

		ylim = np.max(np.abs(np.append(ax5.get_ylim(), ax6.get_ylim())))
		ax5.set_ylim(-ylim, ylim)
		ax6.set_ylim(-ylim, ylim)

		ax7 = fig.add_subplot(427)
		plot_waveform(ax7, sig_1_full, crop_1, time_units=time_units)
		plt.setp(ax7.get_yticklabels(), visible=False)
		plt.setp(ax7.get_yticklines(), visible=False)

		ax8 = fig.add_subplot(428, sharey=ax7)
		plot_waveform(ax8, sig_2_full, crop_2, time_units=time_units)
		plt.setp(ax8.get_yticklabels(), visible=False)
		plt.setp(ax8.get_yticklines(), visible=False)

		ax1.set_title(filename_1.split('/')[-1])
		ax2.set_title(filename_2.split('/')[-1])

		del_12 = mean_2 - mean_1
		del_34 = mean_4 - mean_3

		ax1.set_ylabel(
			'\n \n ref: ' + filename_1.split('/')[-1].split('.')[0] + ' \n \n $\Delta$: {:.3f}'.format(del_12),
			rotation=0, size='large', labelpad=50)
		ax3.set_ylabel(
			'\n \n ref: ' + filename_2.split('/')[-1].split('.')[0] + ' \n \n $\Delta$: {:.3f}'.format(del_34),
			rotation=0, size='large', labelpad=50)

		plt.savefig(out_filename)

		plt.close(fig)

	filename_1, filename_2, out_filename, filt_params = args
	time_units = kwargs['time_units']
	num_windows = kwargs['num_windows']
	PRF_res = kwargs['PRF_res']

	sigs_full, crops, sigs, refs, dists = dists_compare(*args, **kwargs)

	plot()

	base_filename_1 = filename_1.split('/')[-1].split('.')[0]
	base_filename_2 = filename_2.split('/')[-1].split('.')[0]
	out_fname_1 = 'output/PRFCompare/mean/' + base_filename_1 + '_mean_PRF.png'
	out_fname_2 = 'output/PRFCompare/mean/' + base_filename_2 + '_mean_PRF.png'
	ref_func_1, ref_func_2 = refs
	make_PRF_plot(ref_func_1, out_fname_1, params=filt_params, in_filename='MEAN: ' + base_filename_1, PRF_res=PRF_res)
	make_PRF_plot(ref_func_2, out_fname_2, params=filt_params, in_filename='MEAN: ' + base_filename_2, PRF_res=PRF_res)



def plot_clusters(*args, **kwargs):

	filename_1, filename_2, out_filename,filt_params = args

	sigs_full, crops, sigs, refs, dists = dists_compare(*args, **kwargs)


	def plot():

		def add_filename_table(ax, filenames):
			ax.axis('off')
			filenames = [f.split('/')[-1] for f in filenames]  # remove leading "datasets/"
			arr = [
				['A', filenames[0]],
				['B', filenames[1]]
			]

			title_table = ax.table(
				cellText=arr,
				bbox=[0, 0, 1, 1],
				cellLoc='center',
				# rowColours=['C0', 'C1'],
				colWidths=[.5, 1]
			)

		print 'plotting clusters...'
		d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists

		fig = plt.figure(figsize=(10, 6), tight_layout=True)
		fname_ax = plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
		params_ax = plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
		plot_ax = plt.subplot2grid((6, 10), (0, 4), rowspan=6, colspan=6)

		add_filename_table(fname_ax, [filename_1, filename_2])
		add_filt_params_table(params_ax, filt_params)

		plot_ax.set_aspect('equal')
		plot_ax.set_xlabel('distance to A')
		plot_ax.set_ylabel('distance to B')

		A = [d_1_vs_1, d_1_vs_2]
		B = [d_2_vs_1, d_2_vs_2]
		plot_ax.scatter(*A, c='C0', label='A')
		plot_ax.scatter(*B, c='C1', label='B')

		plot_ax.legend()

		max_lim = np.max([plot_ax.get_xlim()[1], plot_ax.get_ylim()[1]])

		plot_ax.set_xlim([0, max_lim])
		plot_ax.set_ylim([0, max_lim])

		fig.savefig(out_filename)

	plot()



def plot_variance(
		filename,
		out_filename,
		filt_params,
		vary_param_1,
		vary_param_2,

		legend_labels=None,
		load_saved_filts=False,

		time_units='seconds',

		crop=(100, 1100),

		num_windows=5,
		window_size=None,

		tau=.001,

		normalize_volume=True,

		PRF_res=50, 		 # number of divisions used for PRF
		dist_scale='none',	 # 'none', 'a', or 'a + b'
		metric='L2', 		 # 'L1' (abs) or 'L2' (euclidean)
		weight_func=lambda i, j: 1,

		see_samples=5,
		quiet=True,

		annot_hm=False
):

	def plot_trajectory(sig):
		print 'plotting trajectory...'
		fig = plt.figure(figsize=(7, 7))
		ax = fig.add_subplot(111)
		cbar = ax.scatter(sig[:,0], sig[:,1], s=.05, c=np.arange(sig.shape[0]))
		fig.colorbar(cbar)
		fig.savefig('output/PRFCompare/variance/trajectory.png')


	def plot_heatmaps(data_arr):
		print 'plotting heatmaps...'

		out_dir = 'output/PRFCompare/variance/heatmaps/'

		if not clear_dir(out_dir):
			print 'skipping heatmaps'
			return


		def make_hmap_fig(hmap_data):
			fig = plt.figure(figsize=(13, 5), tight_layout=True)

			ax1 = fig.add_subplot(131)
			ax2 = fig.add_subplot(132)
			ax3 = fig.add_subplot(133)


			div1 = make_axes_locatable(ax1)
			div2 = make_axes_locatable(ax2)
			div3 = make_axes_locatable(ax3)
			#
			cax1 = div1.append_axes('right', size='10%', pad=.2)
			cax2 = div2.append_axes('right', size='10%', pad=.2)
			cax3 = div3.append_axes('right', size='10%', pad=.2)

			x = y = np.linspace(0, np.power(2, .5), PRF_res)

			plot_heatmap(ax1, cax1, x, y, hmap_data.pointwise_mean, annot=annot_hm)
			plot_heatmap(ax2, cax2, x, y, hmap_data.pointwise_var, annot=annot_hm)
			plot_heatmap(ax3, cax3, x, y, hmap_data.functional_COV, annot=annot_hm)

			ax1.set_title('pointwise mean')
			ax2.set_title('pointwise variance')
			ax3.set_title('functional COV')

			ticks = np.linspace(0, 1.4, PRF_res, endpoint=True)
			for ax in [ax1, ax2, ax3]:
				ax.xaxis.set_ticks(ticks)
				ax.yaxis.set_ticks(ticks)

			return fig


		if vary_param_2:
			for i, val_2 in enumerate(vary_param_2[1]):
				for j, val_1 in enumerate(vary_param_1[1]):
					data = data_arr[i, j]
					fig = make_hmap_fig(data)
					fig.suptitle(filename.split('/')[-1])
					if legend_labels:
						val_2 = legend_labels[i]
					fname = '{}_{}__{}_{}.png'.format(vary_param_2[0], val_2, vary_param_1[0], val_1)
					fig.savefig(out_dir + fname)
					plt.close(fig)

		else:
			for j, val_1 in enumerate(vary_param_1[1]):
				data = data_arr[j]
				fig = make_hmap_fig(data)
				fig.suptitle(filename.split('/')[-1])
				fname = '{}_{}.png'.format(vary_param_1[0], val_1)
				fig.savefig(out_dir + fname)
				plt.close(fig)


	def show_samples(filt_evo_array):
		print 'plotting filtration movies, PDs and PRFs...'
		base_name = filename.split('/')[-1].split('.')[0]

		dir = 'output/PRFCompare/variance/see_samples/{}/'.format(base_name)

		if os.path.exists(dir):
			r = raw_input('Overwrite {} (y/n)? '.format(dir))
			if r == 'y':
				pass
			else:
				print 'Goodbye'
				sys.exit()

			shutil.rmtree(dir)
		os.makedirs(dir)

		if vary_param_2 and vary_param_2[0] in filt_params:
			for i, val_2 in enumerate(vary_param_2[1]):
				for j, val_1 in enumerate(vary_param_1[1]):

					filt_evo = filt_evo_array[i, j]

					print_title('vary_param_1 : {} \t vary_param_2: {}'.format(val_1, val_2))

					for i, filt in enumerate(filt_evo[::see_samples]):

						worm_num = i * see_samples
						comp_name = '{}_{}__{}_{}_#{}'.format(vary_param_1[0], val_1, vary_param_2[0], val_2, worm_num)
						PD_filename = dir + comp_name + 'PD.png'
						PRF_filename = dir + comp_name + 'PRF.png'
						movie_filename = dir + comp_name + 'movie.mp4'

						make_PD(filt, PD_filename)
						make_PRF_plot(filt, PRF_filename, PRF_res=PRF_res)
						make_movie(filt, movie_filename)

		else:
			for j, val_1 in enumerate(vary_param_1[1]):

				filt_evo = filt_evo_array[j]

				print_title('vary_param_1 : {} '.format(val_1))

				for i, filt in enumerate(filt_evo[::see_samples]):
					worm_num = i * see_samples
					comp_name = '{}_{}_#{}'.format(vary_param_1[0], val_1, worm_num)
					PD_filename = dir + comp_name + 'PD.png'
					PRF_filename = dir + comp_name + 'PRF.png'
					movie_filename = dir + comp_name + 'movie.mp4'

					make_PD(filt, PD_filename)
					make_PRF_plot(filt, PRF_filename, PRF_res=PRF_res)
					make_movie(filt, movie_filename)


	def make_main_fig(data, out_filename):
		print 'plotting variance curves...'
		fig = plt.figure(figsize=(14, 8), tight_layout=True)

		label_kwargs = {
			'rotation': 0,
			'ha': 'right',
			'va': 'center',
			'labelpad': 10,
		}

		fname_ax =  plt.subplot2grid((5, 9), (0, 0), rowspan=1, colspan=2)
		params_ax = plt.subplot2grid((5, 9), (1, 0), rowspan=3, colspan=2)

		ax1 = plt.subplot2grid((5, 9), (0, 3), colspan=6)
		ax2 = plt.subplot2grid((5, 9), (1, 3), colspan=6, sharex=ax1)
		ax3 = plt.subplot2grid((5, 9), (2, 3), colspan=6, sharex=ax1)
		ax4 = plt.subplot2grid((5, 9), (3, 3), colspan=6, sharex=ax1)
		ax5 = plt.subplot2grid((5, 9), (4, 3), colspan=6, sharex=ax1)

		add_filenames_table(fname_ax, [filename, out_filename])
		add_filt_params_table(params_ax, filt_params)

		ax1.set_ylabel('norm of\npointwise\nmean', **label_kwargs)
		ax2.set_ylabel('variance', **label_kwargs)
		ax3.set_ylabel('scaled\nvariance', **label_kwargs)
		ax4.set_ylabel('norm of\npointwise\nvariance', **label_kwargs)
		ax5.set_ylabel('norm of\nfunctional\nCOV', **label_kwargs)

		ax5.set_xlabel(vary_param_1[0])


		def plot_stats_curves(var_data):

			x = vary_param_1[1]
			l, = ax1.plot(x, var_data.pointwise_mean_norm, '--o')
			ax2.plot(x, var_data.variance, '--o')
			ax3.plot(x, var_data.scaled_variance, '--o')
			ax4.plot(x, var_data.pointwise_variance_norm, '--o')
			ax5.plot(x, var_data.functional_COV_norm, '--o')

			return l		# for legend


		if vary_param_2:
			if legend_labels: label_list = legend_labels
			else: label_list = [vary_param_2[0] + ' = ' + str(val) for val in vary_param_2[1]]
			line_list = []
			for i, var_data in enumerate(data):
				l = plot_stats_curves(var_data)
				line_list.append(l)
			fig.legend(line_list, label_list, 'lower left', borderaxespad=3, borderpad=1)

		else:
			plot_stats_curves(data)

		for ax in [ax1, ax2, ax3, ax4, ax5]:
			ax.grid()
			ax.set_ylim(bottom=0)

			if ax is not ax5:
				plt.setp(ax.get_xticklabels(), visible=False)
				plt.setp(ax.get_xticklines(), visible=False)

		fig.savefig(out_filename)

	# ===========================================================================
	# 		MAIN: plot_variance()
	# ===========================================================================

	# options = [PRF_res, time_units, normalize_volume, mean_samp_num, num_windows, window_size, see_samples]

	kwargs = locals()


	from Data import get_variance_data, process_variance_data


	# plot_trajectory(sig)

	prf_evo_array, filt_evo_array = get_variance_data(filename, kwargs)
	stats_data, hmap_data = process_variance_data(prf_evo_array, metric, weight_func, dist_scale, vary_param_2)
	make_main_fig(stats_data, out_filename)
	plot_heatmaps(hmap_data)
	if see_samples: show_samples(filt_evo_array)