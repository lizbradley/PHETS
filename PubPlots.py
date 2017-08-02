import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as pyplot, collections, pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from DCE.Plots import plot_waveform_zoom
from DCE.Tools import auto_crop

from config import WAV_SAMPLE_RATE


def plot_PD_pub(filtration, out_filename):
	def add_persistence_plot(ax, filtration):
		ax.set_aspect('equal')
		min_lim = 0
		max_lim = np.max(filtration.epsilons)
		ax.set_xlim(min_lim, max_lim)
		ax.set_ylim(min_lim, max_lim)

		ax.set_xlabel('birth ($\epsilon$)')
		ax.set_ylabel('death ($\epsilon$)')

		ax.grid(which=u'major', zorder=0)
		ax.minorticks_on()

		ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k')  # diagonal line

		# normal #
		min_size = 0
		t_ms_scale = 50
		p_ms_scale = 30
		color = 'C1'

		# BIG for IDA paper #
		# min_size = 300
		# t_ms_scale = 150
		# p_ms_scale = 60
		# color = 'red'


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


	# IDA paper figures #
	# title_block.tick_params(labelsize=23)
	# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	# xlims = ax.get_xlim()
	# ax.set_xticks([0, round(xlims[1]/2., 4), xlims[1]])
	# ylims = ax.get_ylim()
	# ax.set_yticks([round(ylims[1]/2., 4), ylims[1]])
	# ax.tick_params(labelsize=23)



	fig = pyplot.figure(figsize=(6, 6), dpi=500)
	ax = fig.add_subplot(111)
	add_persistence_plot(ax, filtration)
	pyplot.savefig(out_filename)
	pyplot.close(fig)



def plot_filtration_pub(
		filtration, i, out_filename,

		landmark_size=10,
		landmark_color='lime',

		alpha=1,
		dpi=600
):

	def plot_witnesses(subplot, attractor_data):
		attractor_data = np.array(attractor_data)
		x = attractor_data[:, 0]
		y = attractor_data[:, 1]
		subplot.scatter(
			x, y,
			color='black',
			marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'),
			facecolor='black',
			s=.1)


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
	fig = plt.figure(figsize=(6, 6), dpi=700)
	ax = fig.add_subplot(111)
	plot_witnesses(ax, filtration.witness_coords)
	plot_landmarks(ax, filtration.landmark_coords)
	plot_complex(ax, filtration.get_complex_plot_data(), i)
	eps = [0] + filtration.epsilons
	ax.set_title('$\epsilon = {:.7f}$'.format(eps[i]))

	ax.text(.9, .9, '(a)',
			horizontalalignment='center',
			transform=ax.transAxes)


	plt.savefig(out_filename)
	plt.close(fig)


def plot_dce_pub(traj, out_fname):
	def plot_dce(ax, in_file_name):
		# print 'plotting dce...'

		if isinstance(in_file_name, basestring):
			dce_data = np.loadtxt(in_file_name)
		else:
			dce_data = np.asarray(in_file_name)

		amb_dim = dce_data.shape[1]

		if amb_dim == 2:

			x = dce_data[:, 0]
			y = dce_data[:, 1]
			ax.scatter(x, y, color='black', s=.1)
			# subplot.set_aspect('equal')
			ax.set(adjustable='box-forced', aspect='equal')

			ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
			ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))


		elif amb_dim == 3:
			x = dce_data[:, 0]
			y = dce_data[:, 1]
			z = dce_data[:, 2]

			ax.scatter(x, y, z, color='black', s=.1)
			# subplot.set_aspect('equal')
			ax.set(adjustable='box-forced', aspect='equal')
		# ax.axis('off')


		else:
			print 'ERROR: invalid amb_dim (m)'
			sys.exit()


	fig = pyplot.figure(figsize=(6, 6))

	if traj.shape[1] == 3:
		ax = fig.add_subplot(111, projection='3d')
	else:
		ax = fig.add_subplot(111)

	plot_dce(ax, traj)
	pyplot.savefig(out_fname)
	pyplot.close(fig)


def plot_waveform_sec(
		in_filename,
		out_filename,
		crop=None,


	):
	sig = np.loadtxt(in_filename)

	fig = pyplot.figure(figsize=(6, 2), dpi=500, tight_layout=False)

	ax = fig.add_subplot(1, 1, 1)

	if crop:
		crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		sig = sig[crop[0] : crop[1]]
		t = np.linspace(crop[0], crop[1], num=len(sig))

	else:
		t = np.true_divide(np.arange(0, len(sig)), WAV_SAMPLE_RATE)

	ax.plot(t, sig, c='k', lw=.1)

	pyplot.savefig(out_filename), WAV_SAMPLE_RATE