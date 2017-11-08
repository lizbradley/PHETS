import numpy
import numpy as np
from matplotlib import pyplot as plt

import signals, PH
from PH import make_PD, make_PRF_plot, make_movie
from PH.TitleBox import filt_params_table
from utilities import print_title


def dists_to_ref_fig(base_filename, i_ref, i_arr, dists, out_filename):
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111)
	ax.plot(i_arr, dists)
	ax.axvline(x=i_ref, linestyle='--', color='k')
	ax.set_xlabel('i (filename)')
	# ax.set_ylabel('$distance \quad ({\epsilon}^2 \; \# \; holes)$')
	ax.set_ylabel('$distance$')
	ax.xaxis.set_ticks(i_arr[::2])
	ax.grid()
	ax.set_ylim(bottom=0)
	title = ax.set_title(base_filename)
	title.set_position([.5, 1.05])
	plt.savefig(out_filename)
	plt.close(fig)



def samples(filts, interval, dir, vary_param_1=None, vary_param_2=None):

	if vary_param_1 is None and vary_param_2 is None:
		filts_vv = [[filts]]
	elif vary_param_1 is not None and vary_param_2 is None:
		filts_vv = [[fs] for fs in filts]
	else:       # neither are None
		filts_vv = filts

	del filts

	for i, filts_v in enumerate(filts_vv):
		for j, filts in enumerate(filts_v):
			for k, filt in enumerate(filts[::interval]):
				base_name = '{}/{}__'.format(dir, filt.name)

				if vary_param_1 is not None:
					base_name = '{}{}_{}__'.format(
						base_name,
						vary_param_1[0], vary_param_1[1][i]
					)

				if vary_param_2 is not None:
					base_name = '{}{}_{}__'.format(
						base_name,
						vary_param_2[0], vary_param_2[1][j]
					)

				print_title(base_name.split('/')[-1][:-2])

				PD_filename = base_name + 'PD.png'
				PRF_filename = base_name + 'PRF.png'
				movie_filename = base_name + 'movie.mp4'

				make_PD(filt, PD_filename)
				make_PRF_plot(filt, PRF_filename)
				make_movie(filt, movie_filename)


def plot_dists_ax(ax, d, mean, traj):
	crop = traj.crop_lim
	num_windows = traj.num_windows
	t = np.linspace(crop[0], crop[1], num_windows, endpoint=False)
	ticks = np.linspace(crop[0], crop[1], num_windows + 1, endpoint=True)
	offset = (t[1] - t[0]) / 2
	ax.plot(t + offset, d, marker='o', linestyle='None', ms=10, zorder=3)
	ax.axhline(y=mean, linestyle='--', color='forestgreen', lw=2)
	ax.grid(axis='x')
	ax.set_xticks(ticks)


def dists_to_means_fig(refs, dists, traj1, traj2, time_units, out_filename):
	print 'plotting distances...'

	sig_1 = traj1.project()
	sig_2 = traj2.project()

	d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists
	mean_prf_1, mean_prf_2 = refs
	fig = plt.figure(figsize=(18, 8), tight_layout=True)

	mean_11 = np.mean(d_1_vs_1)
	mean_21 = np.mean(d_2_vs_1)
	mean_12 = np.mean(d_1_vs_2)
	mean_22 = np.mean(d_2_vs_2)

	import matplotlib.gridspec as gridspec

	gs = gridspec.GridSpec(5, 5)
	# row 1
	ax1 = plt.subplot(gs[0:2,   0:2])
	ax2 = plt.subplot(gs[0:2,   2:4])
	ax3 = plt.subplot(gs[0:2,   4  ])

	# row 2
	ax4 = plt.subplot(gs[2:4,   0:2])
	ax5 = plt.subplot(gs[2:4,   2:4])
	ax6 = plt.subplot(gs[2:4,   4  ])

	# row 3
	ax7 = plt.subplot(gs[4,     0:2])
	ax8 = plt.subplot(gs[4,     2:4])

	plot_dists_ax(ax1, d_1_vs_1, mean_11, traj1)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklines(), visible=False)
	ax1.set_ylim(bottom=0)

	plot_dists_ax(ax2, d_2_vs_1, mean_21, traj2)
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklines(), visible=False)

	PH.Plots.PRF_ax(mean_prf_1, ax3, annot_hm=True)

	plot_dists_ax(ax4, d_1_vs_2, mean_12, traj1)
	plt.setp(ax4.get_xticklabels(), visible=False)
	plt.setp(ax4.get_xticklines(), visible=False)

	plot_dists_ax(ax5, d_2_vs_2, mean_22, traj2)
	plt.setp(ax5.get_yticklabels(), visible=False)
	plt.setp(ax5.get_xticklabels(), visible=False)
	plt.setp(ax5.get_xticklines(), visible=False)

	PH.Plots.PRF_ax(mean_prf_2, ax6, annot_hm=True)

	signals.plots.ts_zoom(ax7, sig_1)
	ax7.grid(axis='x', zorder=0)
	ax7.set_xlim(left=0)

	signals.plots.ts_zoom(ax8, sig_2)
	ax8.grid(axis='x', zorder=0)
	ax8.set_xlim(left=0)

	ylim = np.max(np.abs(np.append(ax7.get_ylim(), ax8.get_ylim())))
	ax7.set_ylim(-ylim, ylim)
	ax8.set_ylim(-ylim, ylim)

	ax1.set_title(traj1.name)
	ax2.set_title(traj2.name)

	del_12 = mean_21 - mean_11
	del_34 = mean_22 - mean_12

	row_label = '\n \n ref: {} \n \n $\Delta$: {:.3f}'
	ax1.set_ylabel(
		row_label.format(traj1.name, del_12),
		rotation=90, size='large', labelpad=10
	)
	ax4.set_ylabel(
		row_label.format(traj2.name, del_34),
		rotation=90, size='large', labelpad=10
	)

	plt.savefig(out_filename)
	plt.close(fig)


def clusters_fig(dists, filt_params, fname1, fname2, out_fname):

	def legend(ax, filenames):
		ax.axis('off')
		arr = [['A', filenames[0]], ['B', filenames[1]]]
		ax.table(
			cellText=arr,
			bbox=[0, 0, 1, 1],
			cellLoc='center',
			colWidths=[.5, 1]
		)

	d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists

	fig = plt.figure(figsize=(10, 6), tight_layout=True)
	fname_ax =  plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	params_ax = plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax =   plt.subplot2grid((6, 10), (0, 4), rowspan=6, colspan=6)

	legend(fname_ax, [fname1, fname2])
	filt_params_table(params_ax, filt_params)

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

	fig.savefig(out_fname)


def roc_ax(ax, data, k, title):
	fpr, tpr = data
	l, = ax.plot(fpr, tpr, clip_on=False, lw=3, zorder=0)
	k = np.arange(*k)

	ax.plot([0, 1], [0, 1], '--', c='k')

	def sparsify(data, k):
		data = np.asarray(data)
		k_args = (k % 0.5 == 0.0)
		# k_args = (k % 1.0 == 0.0)
		k_pts = k[k_args]
		data_pts = data[:, k_args]
		return data_pts, k_pts

	data_sp, k_sp = sparsify(data, k)

	fpr_sp, tpr_sp = data_sp

	cm = ax.scatter(fpr_sp, tpr_sp, s=150, zorder=10, clip_on=False, c=k_sp)

	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	ax.grid()
	ax.set_aspect('equal')
	ax.set_xlabel('false positive rate')
	ax.set_ylabel('true positive rate')
	ax.set_title(title, y=1.02)
	return l, cm


def dual_roc_fig(data, k, label_1, label_2, fname, vary_param):
	fig = plt.figure(figsize=(10, 4),dpi=100)
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.92, 0.05, 0.03, 0.9])

	lines = []

	for data_wl in data:
		prf_data_1, prf_data_2 = data_wl

		roc_ax(ax1, prf_data_1, k, 'is it a {}?'.format(label_1))
		l, cm = roc_ax(ax2, prf_data_2, k, 'is it a {}?'.format(label_2))
		lines.append(l)

	bounds = np.arange(k[0], k[1] + .5, .5)
	# bounds = np.arange(k[0], k[1] + 1, 1)
	cb = fig.colorbar(cm, cax=cax, boundaries=bounds)
	cb.set_label("$k$", labelpad=-1, size=19)

	labels = bounds[::2]
	loc = labels + .25
	# loc = labels + .5
	cb.set_ticks(loc)
	cb.set_ticklabels([int(l) for l in labels])
	cb.ax.tick_params(labelsize=14)

	# fig.suptitle('k = range({}, {}, {})'.format(*k), fontsize=16)
	if vary_param is not None:
		fig.legend(lines, vary_param[1], loc=3)
		fig.suptitle('varying parameter '+vary_param[0])
		fig.subplots_adjust(top = 0.85)

	plt.savefig(fname)


def plot_weight_functions(vary_param_2, weight_func, filt_params):
	print 'plotting weight function(s)...'
	dir = 'output/PRFCompare/variance/weight_functions/'
	clear_temp_files(dir)

	if vary_param_2 and vary_param_2[0] == 'weight_func':
		funcs = vary_param_2[1]
		fnames = legend_labels

	else:
		funcs = [weight_func]
		fnames = ['f']

	for fname, func in zip(fnames, funcs):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		div = make_axes_locatable(ax)
		cax = div.append_axes('right', size='10%', pad=.2)

		x = y = np.linspace(0, 2 ** .5, filt_params['num_divisions'])
		xx, yy = np.meshgrid(x, y)
		z = func(xx, yy)
		if isinstance(z, int):
			z = xx * 0 + z

		mask = lambda x, y: x > y
		mask = mask(xx, yy)
		mask = np.where(mask == True, np.nan, 1)
		z = np.multiply(z, mask)

		plot_heatmap(ax, cax, x, y, z)
		plt.savefig('{}{}.png'.format(dir, fname))


def plot_heatmaps(data_arr, data_arr_pre_weight, filt_params, vary_param_1,
                  vary_param_2, annot_hm):

	out_dir = 'output/PRFCompare/variance/heatmaps/'

	if not clear_dir(out_dir):
		print 'skipping heatmaps'
		return

	print 'plotting heatmaps...'

	def make_hmap_fig(hmap_data, hmap_data_pw):

		fig = plt.figure(figsize=(12, 8))


		ax1 = fig.add_subplot(231)
		ax2 = fig.add_subplot(232)
		ax3 = fig.add_subplot(233)
		ax4 = fig.add_subplot(234)
		ax5 = fig.add_subplot(235)
		ax6 = fig.add_subplot(236)

		cax = fig.add_axes([.935, .1, .025, .78])

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fig.tight_layout(pad=3, rect=(.05, 0, .95, .95))


		x = y = np.linspace(0, np.power(2, .5), filt_params['num_divisions'])

		plot_heatmap(ax1, cax, x, y, hmap_data.pointwise_mean, annot=annot_hm)
		plot_heatmap(ax2, cax, x, y, hmap_data.pointwise_var, annot=annot_hm)
		plot_heatmap(ax3, cax, x, y, hmap_data.functional_COV, annot=annot_hm)
		plot_heatmap(ax4, cax, x, y, hmap_data_pw.pointwise_mean, annot=annot_hm)
		plot_heatmap(ax5, cax, x, y, hmap_data_pw.pointwise_var, annot=annot_hm)
		plot_heatmap(ax6, cax, x, y, hmap_data_pw.functional_COV, annot=annot_hm)


		ax1.set_title('pointwise mean',		fontsize=12, y=1.05)
		ax2.set_title('pointwise variance', fontsize=12, y=1.05)
		ax3.set_title('functional COV', 	fontsize=12, y=1.05)

		ax1.set_ylabel('weighted',		fontsize=12, labelpad=10)		# abuse y axis label
		ax4.set_ylabel('unweighted',	fontsize=12, labelpad=10)

		ticks = np.linspace(0, 1.4, filt_params['num_divisions'], endpoint=True)
		while len(ticks) > 6:
			ticks = ticks[1::2]
		for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
			ax.xaxis.set_ticks(ticks)
			ax.yaxis.set_ticks(ticks)

		return fig


	if vary_param_2:
		for i, val_2 in enumerate(vary_param_2[1]):
			for j, val_1 in enumerate(vary_param_1[1]):
				data = data_arr[i, j]
				if vary_param_2[0] == 'weight_func':
					data_pw = data_arr_pre_weight[0, j]
				else:
					data_pw = data_arr_pre_weight[i, j]
				fig = make_hmap_fig(data, data_pw)
				fig.suptitle(filename.split('/')[-1])
				if legend_labels:
					val_2 = legend_labels[i]
				fname = '{}_{}__{}_{}.png'.format(
					vary_param_2[0], val_2, vary_param_1[0], val_1
				)
				fig.savefig(out_dir + fname)
				plt.close(fig)

	else:
		data_arr = data_arr[0]
		data_arr_pre_weight = data_arr_pre_weight[0]
		for j, val_1 in enumerate(vary_param_1[1]):
			data = data_arr[j]
			data_pw = data_arr_pre_weight[j]
			fig = make_hmap_fig(data, data_pw)
			fig.suptitle(filename.split('/')[-1])
			fname = '{}_{}.png'.format(vary_param_1[0], val_1)
			fig.savefig(out_dir + fname)
			plt.close(fig)


def plot_variane_fig(data, filt_params, vary_param_1, vary_param_2,
                     out_filename):
	print 'plotting variance curves...'
	fig = plt.figure(figsize=(14, 8), tight_layout=True)

	label_kwargs = {
		'rotation': 0,
		'ha': 'right',
		'va': 'center',
		'labelpad': 10,
	}

	fname_ax =  plt.subplot2grid((5, 9), (0, 0), rowspan=1, colspan=3)
	params_ax = plt.subplot2grid((5, 9), (1, 0), rowspan=3, colspan=3)

	ax1 = plt.subplot2grid((5, 9), (0, 3), colspan=6)
	ax2 = plt.subplot2grid((5, 9), (1, 3), colspan=6, sharex=ax1)
	ax3 = plt.subplot2grid((5, 9), (2, 3), colspan=6, sharex=ax1)
	ax4 = plt.subplot2grid((5, 9), (3, 3), colspan=6, sharex=ax1, sharey=ax2)
	ax5 = plt.subplot2grid((5, 9), (4, 3), colspan=6, sharex=ax1)

	filenames_table(fname_ax, [filename, out_filename])
	filt_params_table(params_ax, filt_params)

	ax1.set_ylabel('norm of mean', **label_kwargs)
	ax2.set_ylabel('global variance', **label_kwargs)
	ax3.set_ylabel('global fano factor', **label_kwargs)
	ax4.set_ylabel('local variance', **label_kwargs)
	ax5.set_ylabel('local fano factor', **label_kwargs)

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
		plot_stats_curves(data[0])

	for ax in [ax1, ax2, ax3, ax4, ax5]:
		ax.grid()
		ax.set_ylim(bottom=0)

		if ax is not ax5:
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_xticklines(), visible=False)

	fig.savefig(out_filename)