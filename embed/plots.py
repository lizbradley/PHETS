import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import signals
from embed.titlebox import slide_window_title, vary_tau_title, \
	compare_vary_tau_title


def traj_ax(ax, data):
	""" plot trajectory `data` to `ax`"""
	ax.scatter(*data.T, color='black', s=.1)
	ax.set(aspect='equal', adjustable='datalim', anchor='C')

	# choose second outermost auto ticks
	yticks = ax.get_yticks()
	ymin, ymax = yticks[1], yticks[-2]
	xticks = ax.get_xticks()
	xmin, xmax = xticks[1], xticks[-2]
	ax.set_yticks([ymin, ymax])
	ax.set_xticks([xmin, xmax])

	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


def slide_window_frame(traj, window, out_fname):
	fig = plt.figure(figsize=(8.5, 6), tight_layout=True, dpi=100)
	gs = gridspec.GridSpec(8, 10)

	fname_ax =        fig.add_subplot(gs[0:2,    :4])
	param_ax =        fig.add_subplot(gs[2:5,    :4])
	ts_ax =           fig.add_subplot(gs[6:8,   :10])
	if traj.dim == 2:
		dce_ax =      fig.add_subplot(gs[0:6,  4:10])
	else: dce_ax =    fig.add_subplot(gs[0:6,  4:10], projection='3d')

	slide_window_title(fname_ax, param_ax, traj, window)
	traj_ax(dce_ax, traj.windows[window].data)
	signals.plots.ts_ax(ts_ax, traj.source_ts, window)
	plt.savefig(out_fname)
	plt.close(fig)


def vary_tau_frame(traj, out_fname):
	fig = plt.figure(figsize=(8.5, 6), tight_layout=True, dpi=100)

	gs = gridspec.GridSpec(8, 10)

	fname_ax =        fig.add_subplot(gs[0:1,    :4])
	param_ax =        fig.add_subplot(gs[2:5,    :4])
	ts_ax =           fig.add_subplot(gs[6:8,   :10])
	if traj.dim == 2:
		dce_ax =      fig.add_subplot(gs[0:6,  4:10])
	else: dce_ax =    fig.add_subplot(gs[0:6,  4:10], projection='3d')

	vary_tau_title(fname_ax, param_ax, traj)
	traj_ax(dce_ax, traj.data)
	signals.plots.ts_ax(ts_ax, traj.source_ts)
	plt.savefig(out_fname)
	plt.close(fig)


def compare_frame(traj1, traj2, out_fname, tau):
	fig = plt.figure(figsize=(10, 5), dpi=100)
	gs = gridspec.GridSpec(8, 16)

	title_ax =         fig.add_subplot(gs[ : ,   :4 ])

	ts1_ax =           fig.add_subplot(gs[7:8,   4:10])
	ts2_ax =           fig.add_subplot(gs[7:8,  10:16])

	if traj1.dim == 2:
		dce1_ax =      fig.add_subplot(gs[0:6,   4:10])
	else: dce1_ax =    fig.add_subplot(gs[0:6,   4:10], projection='3d')

	if traj2.dim == 2:
		dce2_ax =      fig.add_subplot(gs[0:6,  10:16])
	else: dce2_ax =    fig.add_subplot(gs[0:6,   4:10], projection='3d')

	fig.subplots_adjust(
		left=.03, right=.97,
		bottom=.1, top=.95,
		wspace=15, hspace=0
	)

	compare_vary_tau_title(title_ax, traj1, traj2, tau)

	traj_ax(dce1_ax, traj1.data)
	signals.plots.ts_full_ax(ts1_ax, traj1.source_ts)

	traj_ax(dce2_ax, traj2.data)
	signals.plots.ts_full_ax(ts2_ax, traj2.source_ts)

	plt.savefig(out_fname)
	plt.close(fig)


