import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import signals, PH
from PH.plots import heatmap_ax
from PH.titlebox import filt_params_table
from PRFstats.data import is_filt_param
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



def samples(filts, cmd, dir, vary_param_1=None, vary_param_2=None):

	if not (is_filt_param(vary_param_1) or is_filt_param(vary_param_2)):
		filts_vv = [[filts]]
	elif is_filt_param(vary_param_1) and not is_filt_param(vary_param_2):
		filts_vv = [filts]
	else:
		filts_vv = filts

	del filts

	if isinstance(cmd, dict):
		interval = cmd['interval']
		try:
			filt_step = cmd['filt_step']
		except KeyError:
			filt_step = None
	else:
		interval = cmd
		filt_step = None


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
				filt.plot_PD(PD_filename)

				PRF_filename = base_name + 'PRF.png'
				filt.plot_PRF(PRF_filename)

				if filt_step:
					complex_filename = base_name + 'comp.png'
					filt.plot_complex(filt_step, complex_filename)
				else:
					movie_filename = base_name + 'movie.mp4'
					filt.movie(movie_filename)


def dists_ax(ax, d, mean, traj):
	crop = traj.crop_lim
	num_windows = traj.num_windows
	t = np.linspace(crop[0], crop[1], num_windows, endpoint=False)
	ticks = np.linspace(crop[0], crop[1], num_windows + 1, endpoint=True)
	offset = (t[1] - t[0]) / 2
	ax.plot(t + offset, d, marker='o', linestyle='None', ms=10, zorder=3)
	ax.axhline(y=mean, linestyle='--', color='forestgreen', lw=2)
	ax.grid(axis='x')
	ax.set_xticks(ticks)


def dists_to_means_fig(refs, dists, traj1, traj2, out_filename):
	print 'plotting distances...'

	ts1 = traj1.project()
	ts2 = traj2.project()

	d_1_vs_1, d_2_vs_1, d_1_vs_2, d_2_vs_2 = dists
	mean_prf_1, mean_prf_2 = refs
	fig = plt.figure(figsize=(18, 8), tight_layout=True)

	mean_11 = np.mean(d_1_vs_1)
	mean_21 = np.mean(d_2_vs_1)
	mean_12 = np.mean(d_1_vs_2)
	mean_22 = np.mean(d_2_vs_2)


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

	dists_ax(ax1, d_1_vs_1, mean_11, traj1)
	plt.setp(ax1.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklines(), visible=False)
	ax1.set_ylim(bottom=0)

	dists_ax(ax2, d_2_vs_1, mean_21, traj2)
	plt.setp(ax2.get_yticklabels(), visible=False)
	plt.setp(ax2.get_xticklabels(), visible=False)
	plt.setp(ax2.get_xticklines(), visible=False)

	PH.plots.PRF_ax(mean_prf_1, ax3, annot_hm=True)

	dists_ax(ax4, d_1_vs_2, mean_12, traj1)
	plt.setp(ax4.get_xticklabels(), visible=False)
	plt.setp(ax4.get_xticklines(), visible=False)

	dists_ax(ax5, d_2_vs_2, mean_22, traj2)
	plt.setp(ax5.get_yticklabels(), visible=False)
	plt.setp(ax5.get_xticklabels(), visible=False)
	plt.setp(ax5.get_xticklines(), visible=False)

	PH.plots.PRF_ax(mean_prf_2, ax6, annot_hm=True)

	signals.plots.ts_crop_ax(ax7, ts1)
	ax7.grid(axis='x', zorder=0)
	# ax7.set_xlim(left=0)

	signals.plots.ts_crop_ax(ax8, ts2)
	ax8.grid(axis='x', zorder=0)
	# ax8.set_xlim(left=0)

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


