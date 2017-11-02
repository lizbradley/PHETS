"""
publication / latex appropriate plotting function duplicates for IDA paper
called by ida_figs
"""
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as pyplot, collections, pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from DCE.Plots import plot_signal_zoom
from DCE.Tools import auto_crop

from config import SAMPLE_RATE

from utilities import normalize_volume


def letter_label(ax, label, nudge_r=0.):
	ax.text(
		.95 + nudge_r, .95, label,
		size='large',
		horizontalalignment='right',
		verticalalignment='top',
		transform=ax.transAxes,
		bbox=dict(
			alpha=1,
			facecolor='white',
			edgecolor='black',
			pad=5,
			boxstyle='round, pad=.5'
		)
	)

def plot_PD_pub(filtration, out_filename, label=None, ticks=None, cbar=True, fig=None):
	def add_persistence_plot(ax, filtration):

		min_lim = 0
		max_lim = np.max(filtration.epsilons)
		ax.set_xlim(min_lim, max_lim)
		ax.set_ylim(min_lim, max_lim)

		ax.set_xlabel('$\epsilon_B$')
		ax.set_ylabel('$\epsilon_D$')

		ax.grid(which=u'major', zorder=0)
		ax.minorticks_on()
		ax.set_aspect('equal')
		ax.set_axisbelow(True)

		ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k', zorder=0)  # diagonal line

		data = filtration.get_PD_data()
		if data == 'empty':
			return


		# count by size #
		#################
		#
		# color = 'C3'
		# t_min = 5
		# p_min = 5
		# t_scale = 15
		# p_scale = 15
		#
		# def msize(n, scale, min):
		# 	return(n * scale) ** 1.2
		#
		# add legend #
		# mark_t_1 = ax.scatter([], [], marker='^', s=msize(1, t_scale, t_min), c=color)
		# mark_t_3 = ax.scatter([], [], marker='^', s=msize(3, t_scale, t_min), c=color)
		# mark_t_5 = ax.scatter([], [], marker='^', s=msize(5, t_scale, t_min), c=color)
		#
		# mark_p_1 = ax.scatter([], [], s=msize(1, p_scale, p_min), c=color)
		# mark_p_3 = ax.scatter([], [], s=msize(3, p_scale, p_min), c=color)
		# mark_p_5 = ax.scatter([], [], s=msize(5, p_scale, p_min), c=color)
		#
		# marks = (mark_t_1, mark_t_3, mark_t_5, mark_p_1, mark_p_3, mark_p_5)
		# labels = ('', '', '', '1', '3', '5')
		#
		# ax.legend(
		# 	marks, labels, loc='lower right', ncol=2, markerscale=1,
		# 	borderpad=1,
		# 	labelspacing=1,
		# 	framealpha=1,
		# 	columnspacing=0,
		# 	borderaxespad=3
		# 	# edgecolor='k'
		# )

		# # end add legend #

		# data = filtration.get_PD_data()
		# if data == 'empty':
		# 	return
		#
		# if len(data.mortal) > 0:
		# 	x_mor, y_mor, count_mor = data.mortal
		# 	ax.scatter(x_mor, y_mor, s=msize(count_mor, p_scale, p_min), c=color, clip_on=True, zorder=100)
		#
		# if len(data.immortal) > 0:
		# 	x_imm, count_imm = data.immortal
		# 	y_imm = [max_lim for i in x_imm]
		# 	ax.scatter(x_imm, y_imm, marker='^', s=msize(count_imm, t_scale, t_min), c=color, clip_on=False, zorder=100)

		#

		pass
		import matplotlib.colorbar as colorbar

		# count by color #
		##################

		sc = None

		if len(data.mortal) > 0:
			x_mor, y_mor, count_mor = data.mortal
			sc = ax.scatter(x_mor, y_mor, s=70,
							c=count_mor, alpha=.8,
							clip_on=True, zorder=100,
							vmin=1, vmax=5)


		if len(data.immortal) > 0:
			x_imm, count_imm = data.immortal

			y_imm = [max_lim for i in x_imm]
			sc = ax.scatter(x_imm, y_imm, marker='^', s=120,
							c=count_imm, alpha=.8,
							clip_on=False, zorder=100,
					 		vmin=1, vmax=5)

		if cbar and sc:
			levels = [1, 2, 3, 4, 5]

			cb = plt.colorbar(sc, extend='max', extendrect=True, extendfrac=.2, ax=ax, values=levels)

			cb.ax.text(1.5, 0.10, '1')
			cb.ax.text(1.5, 0.35, '2')
			cb.ax.text(1.5, 0.60, '3')
			cb.ax.text(1.5, 0.85, '4')
			cb.ax.text(1.5, 1.10, '5+')

	if isinstance(out_filename, basestring):
		fig = pyplot.figure(figsize=(6, 6), dpi=500)
		ax = fig.add_subplot(111)
		add_persistence_plot(ax, filtration)


	else:
		ax = out_filename
		# if cbar is True:
			# divider = make_axes_locatable(ax)
			# cax = divider.append_axes('right', size='5%', pad=0.1)
		add_persistence_plot(ax, filtration)

	ax.text(.75, .5, r'   $\beta_1$   ',
			horizontalalignment='center',
			verticalalignment='center',
			size='x-large',
			bbox=dict(alpha=1, facecolor='white', pad=10),
			transform=ax.transAxes)

	if label:
		letter_label(ax, label)


	if ticks is not None:
		ax.xaxis.set_ticks(ticks)
		ax.yaxis.set_ticks(ticks)
	else:
		ax.xaxis.set_major_locator(MaxNLocator(5))
		ax.yaxis.set_major_locator(MaxNLocator(5))


	if isinstance(out_filename, basestring):
		pyplot.savefig(out_filename)
		pyplot.close(fig)




def plot_filtration_pub(
		filtration, i,
		output,
		landmark_size=15,
		landmark_color='lime',
		line_width=1,
		alpha=1,
		dpi=700,
		show_eps=True,
		label=False,
		ticks=None
):

	def plot_witnesses(subplot, attractor_data):
		attractor_data = np.array(attractor_data)
		print 'witness data shape:', attractor_data.shape
		x = attractor_data[:, 0]
		y = attractor_data[:, 1]
		subplot.scatter(
			x, y,
			color='black',
			s=1,
			facecolor='black', edgecolor='',
			zorder=1
		)


	def plot_landmarks(subplot, landmark_data):
		landmark_data = np.array(landmark_data)
		print 'landmark data shape:', landmark_data.shape

		x = landmark_data[:, 0]
		y = landmark_data[:, 1]
		subplot.scatter(
			x, y,
			marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'),
			s=landmark_size,
			facecolor=landmark_color,
			edgecolor='k',
			lw=.3,
			zorder=3
		)


	def plot_complex(subplot, complex_data, i):
		"""plots all complexes for full filtration"""

		for j, simplexes_coords in enumerate(complex_data[:i]):
			# f_color, e_color = 'lightskyblue', 'black'
			f_color, e_color = 'C0', 'black'
			# print 'eps index', j

			simplexes = collections.PolyCollection(
				simplexes_coords,
				edgecolors=e_color,
				facecolors=f_color,
				lw=line_width,
				alpha=alpha,
				zorder=0,
				antialiased=True)

			subplot.add_collection(simplexes)


	print 'plotting filtration frame...'

	if isinstance(output, basestring):
		fig = plt.figure(figsize=(4.5, 4.4), dpi=dpi, tight_layout=True)
		ax = fig.add_subplot(111)
	else:
		ax = output

	plot_witnesses(ax, filtration.witness_coords)
	plot_landmarks(ax, filtration.landmark_coords)
	plot_complex(ax, filtration.get_complex_plot_data(), i)
	eps = [0] + filtration.epsilons

	if show_eps:
		ax.set_title('$\epsilon = {:.7f}$'.format(eps[i]))

	if label:
		letter_label(ax, label)

	if ticks:
		ax.xaxis.set_ticks(ticks)
		ax.yaxis.set_ticks(ticks)
	ax.set_aspect('equal')
	if isinstance(output, basestring):
		plt.savefig(output)
		plt.close(fig)




def plot_waveform_sec(
		ax,
		sig,
		crop=None,
		normalize=False,
		normalize_crop=False,
		label=False,
		yticks=None
	):


	if isinstance(sig, basestring):
		sig = np.loadtxt(sig)

	if normalize: sig = normalize_volume(sig)



	if crop:
		c0, c1 = (np.array(crop) * SAMPLE_RATE).astype(int)
		sig = sig[c0: c1]
		t = np.linspace(crop[0], crop[1], num=len(sig))


	else:
		t = np.true_divide(np.arange(0, len(sig)), SAMPLE_RATE)

	if normalize_crop: sig = normalize_volume(sig)


	ax.plot(t, sig, c='k', lw=.5)


	if label:
		letter_label(ax, label, nudge_r=.03)

	if yticks is not None:
		ax.yaxis.set_ticks(yticks)

	ax.set_xlabel('time (s)')
	ax.set_ylabel('$x(t)$')




def plot_dce_pub(ax, traj, ticks=False, label=False):

	if isinstance(traj, basestring):
		traj = np.loadtxt(traj)

	ax.set_aspect('equal')

	if traj.shape[1] == 2:
		x = traj[:, 0]
		y = traj[:, 1]
		ax.scatter(
			x, y,
			s=.5,
			marker='o',
			facecolor='black', edgecolor=''

		)

	elif traj.shape[1] == 3:

		x = traj[:, 0]
		y = traj[:, 1]
		z = traj[:, 2]
		ax.scatter(x, y, z, color='black', s=.1)

	else:
		print 'ERROR: invalid amb_dim (m)'
		sys.exit()

	if ticks is not None:
		ax.xaxis.set_ticks(ticks)
		ax.yaxis.set_ticks(ticks)

	# if label:
	# 	ax.text(.94, .95, label,
	# 			horizontalalignment='center',
	# 			verticalalignment='center',
	# 			transform=ax.transAxes)

	if label:
		letter_label(ax, label)


	ax.set_xlabel('$x(t)$')
	ax.set_ylabel('$x(t + \\tau)$')

	# ax.set_ylim([-1.1, 1.1])
	# ax.set_xlim([-1.1, 1.1])