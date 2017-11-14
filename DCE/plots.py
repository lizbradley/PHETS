import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import titlebox
import signals


def plot_title(fname_ax, param_ax, traj, window):
	""" for slide window movies """

	titlebox.title_table(fname_ax, traj.name, window)

	params = {
		'tau': traj.embed_params['tau'],
		'm': traj.embed_params['m'],
		'crop': traj.crop_lim,
		'time_units': traj.time_units,
		'window_length': traj.window_length,
		'window_step': traj.num_windows,
	}

	titlebox.param_table(param_ax, params)


def plot_dce(fig, ax, dce_data):
	amb_dim = dce_data.shape[1]

	if amb_dim == 2:
		x = dce_data[:, 0]
		y = dce_data[:, 1]
		ax.scatter(x, y, color='black', s=.1)

	elif amb_dim == 3:
		x = dce_data[:, 0]
		y = dce_data[:, 1]
		z = dce_data[:, 2]
		ax.scatter(x, y, z, color='black', s=.1)

	ax.set(aspect='equal', adjustable='datalim', anchor='C')

	# choose second outermost auto ticks
	yticks = ax.get_yticks()
	ymin, ymax = yticks[1], yticks[-2]
	xticks = ax.get_xticks()
	xmin, xmax = xticks[1], xticks[-2]
	ax.set_yticks([ymin, ymax])
	ax.set_xticks([xmin, xmax])


	fig.subplots_adjust(left=.07, bottom=.07, right=.93, top=.93, wspace=.5, hspace=.5)


def slide_window_frame(traj, window, out_fname):
	fig = plt.figure(figsize=(8, 6), tight_layout=False, dpi=100)
	# fig.subplots_adjust(hspace=.5)

	gs = gridspec.GridSpec(8, 10)

	fname_ax =        fig.add_subplot(gs[0,     :3 ])
	param_ax =        fig.add_subplot(gs[1:4,   :3 ])
	ts_ax =           fig.add_subplot(gs[6:8,   :10])
	if traj.dim == 2:
		dce_ax =      fig.add_subplot(gs[0:6,  4:10])
	else: dce_ax =    fig.add_subplot(gs[0:6,  4:10], projection='3d')

	plot_title(fname_ax, param_ax, traj, window)
	plot_dce(fig, dce_ax, traj.windows[window].data)
	signals.plots.ts_crop_ax(ts_ax, traj.source_ts, window)
	plt.savefig(out_fname)
	plt.close(fig)
