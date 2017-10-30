import numpy
import numpy as np
from matplotlib import pyplot as plt

import signals
from PH import make_PD, make_PRF_plot, make_movie
from Utilities import print_title


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
	fig = plt.figure(
		figsize=(10, 4),
		dpi=100
	)

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

	# fig.tight_layout(rect=[0, 0, 1, 0.9])

	plt.savefig(fname)


def samples(filts, interval, dir, vary_param_1=None, vary_param_2=None):

	if vary_param_1 is None and vary_param_2 is None:
		filts = [[filts]]
	elif vary_param_1 is not None and vary_param_2 is None:
		filts = [[fs] for fs in filts]
	elif vary_param_1 is not None and vary_param_2 is not None:
		pass
	else:
		print 'ERROR: vary_param_1 is None, but vary_param_2 is not None'

	for i, filts_v1 in enumerate(filts):
		for j, filts_v2 in enumerate(filts_v1):
			for k, filt in enumerate(filts_v2[::interval]):
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


def plot_dists_ax(ax, d, mean, crop):
	t = np.linspace(crop[0], crop[1], num_windows, endpoint=False)
	ticks = np.linspace(crop[0], crop[1], num_windows + 1, endpoint=True)
	offset = (t[1] - t[0]) / 2
	ax.plot(t + offset, d, marker='o', linestyle='None', ms=10, zorder=3)
	ax.axhline(y=mean, linestyle='--', color='forestgreen', lw=2)
	ax.grid(axis='x')
	ax.set_xticks(ticks)

def dists_vs_means_fig(refs, dists):


	print 'plotting distances...'

	time_units = kwargs['time_units']
	num_windows = kwargs['num_windows']

	filename_1, filename_2, out_filename, filt_params = args

	sig_1_full, sig_2_full = sigs_full
	crop_1, crop_2 = crops
	sig_1, sig_2 = sigs
	d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists

	fig = plt.figure(figsize=(18, 9), tight_layout=True)

	mean_1 = np.mean(d_1_vs_1)
	mean_2 = np.mean(d_2_vs_1)
	mean_3 = np.mean(d_1_vs_2)
	mean_4 = np.mean(d_2_vs_2)


	ax1 = fig.add_subplot(421)
	plot_dists_ax(ax1, d_1_vs_1, mean_1, crop_1)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklines(), visible=False)
	ax1.set_ylim(bottom=0)

	ax2 = fig.add_subplot(422, sharey=ax1)
	plot_dists_ax(ax2, d_2_vs_1, mean_2, crop_2)
	plt.setp(ax2.get_yticklabels(), visible=False)
	# plt.setp(ax2.get_yticklines(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklines(), visible=False)

	ax3 = fig.add_subplot(423, sharey=ax1, sharex=ax1)
	plot_dists_ax(ax3, d_1_vs_2, mean_3, crop_1)
	plt.setp(ax3.get_xticklabels(), visible=False)
	plt.setp(ax3.get_xticklines(), visible=False)

	ax4 = fig.add_subplot(424, sharey=ax1, sharex=ax2)
	plot_dists_ax(ax4, d_2_vs_2, mean_4, crop_2)
	plt.setp(ax4.get_yticklabels(), visible=False)
	# plt.setp(ax4.get_yticklines(), visible=False)

	plt.setp(ax4.get_xticklabels(), visible=False)
	plt.setp(ax4.get_xticklines(), visible=False)

	ax5 = fig.add_subplot(425, sharex=ax1)
	signals.plots.ts_zoom(ax5, None, crop_1, time_units=time_units, sig=sig_1)
	ax5.grid(axis='x', zorder=0)
	plt.setp(ax5.get_yticklabels(), visible=False)
	plt.setp(ax5.get_yticklines(), visible=False)


	ax6 = fig.add_subplot(426, sharex=ax2)
	signals.plots.ts_zoom(ax6, None, crop_2, time_units=time_units, sig=sig_2)
	ax6.grid(axis='x', zorder=0)
	plt.setp(ax6.get_yticklabels(), visible=False)
	plt.setp(ax6.get_yticklines(), visible=False)

	ylim = np.max(np.abs(np.append(ax5.get_ylim(), ax6.get_ylim())))
	ax5.set_ylim(-ylim, ylim)
	ax6.set_ylim(-ylim, ylim)

	ax7 = fig.add_subplot(427)
	signals.plots.ts(ax7, sig_1_full, crop_1, time_units=time_units)
	plt.setp(ax7.get_yticklabels(), visible=False)
	plt.setp(ax7.get_yticklines(), visible=False)

	ax8 = fig.add_subplot(428, sharey=ax7)
	signals.plots.ts(ax8, sig_2_full, crop_2, time_units=time_units)
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