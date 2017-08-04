import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as pyplot, collections, pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from DCE.Plots import plot_signal_zoom
from DCE.Tools import auto_crop

from config import WAV_SAMPLE_RATE

from Tools import normalize_volume



def plot_PD_pub(filtration, out_filename, label=None):
	def add_persistence_plot(ax, filtration):
		ax.set_aspect('equal')
		min_lim = 0
		max_lim = np.max(filtration.epsilons)
		ax.set_xlim(min_lim, max_lim)
		ax.set_ylim(min_lim, max_lim)

		ax.set_xlabel('$\epsilon_B$')
		ax.set_ylabel('$\epsilon_D$')

		ax.grid(which=u'major', zorder=0)
		ax.minorticks_on()

		ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k')  # diagonal line

		# normal #
		min_size = 0
		t_ms_scale = 50
		p_ms_scale = 30
		color = 'C3'

		# add legend #
		mark_t_1 = ax.scatter([], [], marker='^', s=t_ms_scale, c=color)
		mark_t_3 = ax.scatter([], [], marker='^', s=t_ms_scale * 3, c=color)
		mark_t_5 = ax.scatter([], [], marker='^', s=t_ms_scale * 5, c=color)

		mark_p_1 = ax.scatter([], [], s=p_ms_scale, c=color)
		mark_p_3 = ax.scatter([], [], s=p_ms_scale * 3, c=color)
		mark_p_5 = ax.scatter([], [], s=p_ms_scale * 5, c=color)

		marks = (mark_t_1, mark_t_3, mark_t_5, mark_p_1, mark_p_3, mark_p_5)
		labels = ('', '', '', '1', '3', '5')

		ax.legend(
			marks, labels, loc='lower right', ncol=2, markerscale=1,
			borderpad=1,
			labelspacing=1,
			framealpha=1,
			columnspacing=0,
			borderaxespad=3
			# edgecolor='k'
		)

		data = filtration.get_PD_data()
		if data == 'empty':
			return

		if len(data.mortal) > 0:
			x_mor, y_mor, count_mor = data.mortal
			ax.scatter(x_mor, y_mor, s=(count_mor * p_ms_scale) + min_size, clip_on=False, c=color)

		if len(data.immortal) > 0:
			x_imm, count_imm = data.immortal
			y_imm = [max_lim for i in x_imm]
			ax.scatter(x_imm, y_imm, marker='^', s=(count_imm * t_ms_scale) + min_size, c=color, clip_on=False)

		# end add legend #


	if isinstance(out_filename, basestring):
		fig = pyplot.figure(figsize=(6, 6), dpi=500)
		ax = fig.add_subplot(111)
		add_persistence_plot(ax, filtration)

	else:
		ax = out_filename
		add_persistence_plot(ax, filtration)

	ax.text(.5, .8, r'   $\beta_1$   ',
			horizontalalignment='center',
			verticalalignment='center',
			size='x-large',
			bbox=dict(alpha=1, facecolor='white', pad=10),
			transform=ax.transAxes)

	if label:
		ax.text(.7, .95, label,
				horizontalalignment='center',
				verticalalignment='center',
				transform=ax.transAxes)

	if isinstance(out_filename, basestring):
		pyplot.savefig(out_filename)
		pyplot.close(fig)




def plot_filtration_pub(
		filtration, i,
		out_filename,

		landmark_size=10,
		landmark_color='lime',

		alpha=1,
		dpi=700,
		show_eps=True,
		label=None,
		ticks=None
):

	def plot_witnesses(subplot, attractor_data):
		attractor_data = np.array(attractor_data)
		x = attractor_data[:, 0]
		y = attractor_data[:, 1]
		subplot.scatter(
			x, y,
			color='black',
			s=1,
			marker='o',
			facecolor='black', edgecolor=''
		)


	def plot_landmarks(subplot, landmark_data):
		landmark_data = np.array(landmark_data)
		x = landmark_data[:, 0]
		y = landmark_data[:, 1]
		subplot.scatter(
			x, y,
			marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'),
			s=landmark_size,
			facecolor=landmark_color
		)


	def plot_complex(subplot, complex_data, i):
		"""plots all complexes for full filtration"""

		for j, simplexes_coords in enumerate(complex_data[:i]):
			f_color, e_color = 'C0', 'black'
			print 'eps index', j

			simplexes = collections.PolyCollection(
				simplexes_coords,
				edgecolors=e_color,
				facecolors=f_color,
				lw=1,
				alpha=alpha,
				zorder=0,
				animated=True,
				antialiased=True)

			subplot.add_collection(simplexes)




	print 'plotting filtration frame...'

	if isinstance(out_filename, basestring):
		fig = plt.figure(figsize=(4.5, 4.4), dpi=dpi, tight_layout=True)
		ax = fig.add_subplot(111)
	else:
		ax = out_filename
	plot_witnesses(ax, filtration.witness_coords)
	plot_landmarks(ax, filtration.landmark_coords)
	plot_complex(ax, filtration.get_complex_plot_data(), i)
	eps = [0] + filtration.epsilons

	if show_eps:
		ax.set_title('$\epsilon = {:.7f}$'.format(eps[i]))

	if label:
		ax.text(.7, .95, label,
				horizontalalignment='left',
				verticalalignment='center',
				transform=ax.transAxes)
	if ticks:
		ax.xaxis.set_ticks(ticks)
		ax.yaxis.set_ticks(ticks)
	ax.set_aspect('equal')
	if isinstance(out_filename, basestring):
		plt.savefig(out_filename)
		plt.close(fig)




def plot_waveform_sec(
		sig,
		out_filename,
		crop=None,
		normalize=True
	):

	if isinstance(sig, basestring):
		sig = np.loadtxt(sig)

	if normalize: sig = normalize_volume(sig)

	fig = pyplot.figure(figsize=(5, 2), dpi=600, tight_layout=True)

	ax = fig.add_subplot(1, 1, 1)

	if crop:
		c0, c1 = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		sig = sig[c0: c1]
		t = np.linspace(crop[0], crop[1], num=len(sig))

	else:
		t = np.true_divide(np.arange(0, len(sig)), WAV_SAMPLE_RATE)

	ax.plot(t, sig, c='k', lw=.5)

	pyplot.savefig(out_filename), WAV_SAMPLE_RATE



def plot_dce_pub(traj, out_fname):

	fig = pyplot.figure(figsize=(2.3, 2), dpi=600, tight_layout=True)

	if isinstance(traj, basestring):
		traj = np.loadtxt(traj)


	if traj.shape[1] == 2:
		ax = fig.add_subplot(111)
		x = traj[:, 0]
		y = traj[:, 1]
		ax.scatter(
			x, y,
			s=.5,
			marker='o',
			facecolor='black', edgecolor=''

		)

	elif traj.shape[1] == 3:
		ax = fig.add_subplot(111, projection='3d')

		x = traj[:, 0]
		y = traj[:, 1]
		z = traj[:, 2]
		ax.scatter(x, y, z, color='black', s=.1)

	else:
		print 'ERROR: invalid amb_dim (m)'
		sys.exit()

	ax.set_aspect('equal')
	pyplot.savefig(out_fname)
	pyplot.close(fig)


