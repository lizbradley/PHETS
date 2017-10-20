import numpy as np
from matplotlib import pyplot as plt

from PH import make_PD, make_PRF_plot, make_movie
from Utilities import print_title


def plot_roc_ax(ax, data, k, title):
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


def plot_dual_roc_fig(data, k, label_1, label_2, fname, vary_param):
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

		plot_roc_ax(ax1, prf_data_1, k, 'is it a {}?'.format(label_1))
		l, cm = plot_roc_ax(ax2, prf_data_2, k, 'is it a {}?'.format(label_2))
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


def plot_samples(filts, interval, dir, vary_param_1=None, vary_param_2=None):

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
