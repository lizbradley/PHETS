import subprocess
import os
import matplotlib.pyplot as pyplot
import itertools
import numpy as np
from os import system, chdir

from matplotlib import pyplot as plt
from memory_profiler import profile
from sys import platform

from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors

from PH.FiltrationMovie import plot_2D_init, plot_2D_update
from TitleBox import add_filename_table, add_filt_params_table


# from Utilities import mem_profile
from config import MEMORY_PROFILE_ON

from Data import Filtration

f=open("output/run_info/group_by_birth_time_memory.txt","wb")
f2=open("output/run_info/expand_to_2simplexes_memory.txt","wb")
f3=open("output/run_info/build_perseus_in_file_memory.txt","wb")
f4=open("output/run_info/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)



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

	ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k')		# diagonal line


	# normal #
	min_size = 0
	t_ms_scale = 50
	p_ms_scale = 30
	color = 'C0'

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
		#edgecolor='k'
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


# @profile(stream=f4)
def make_PD(filt, out_filename):
	print '\nplotting persistence diagram...'

	fig = pyplot.figure(figsize=(10, 6), tight_layout=True, dpi=300)

	fname_ax = 		pyplot.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	# epsilon_ax = 	pyplot.subplot2grid((6, 10), (1, 0), rowspan=1, colspan=3)
	params_ax = 	pyplot.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		pyplot.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)

	add_persistence_plot(plot_ax, filt)
	add_filename_table(fname_ax, filt.filename)
	add_filt_params_table(params_ax, filt.params)

	# IDA paper figures #
	# title_block.tick_params(labelsize=23)
	# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	# xlims = ax.get_xlim()
	# ax.set_xticks([0, round(xlims[1]/2., 4), xlims[1]])
	# ylims = ax.get_ylim()
	# ax.set_yticks([round(ylims[1]/2., 4), ylims[1]])
	# ax.tick_params(labelsize=23)

	pyplot.savefig(out_filename)
	pyplot.clf()


import matplotlib
import matplotlib.colorbar as colorbar
import numpy.ma as ma


def plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=False):
	plot_ax.set_aspect('equal')

	viridis = matplotlib.cm.get_cmap('viridis')
	colors = [viridis(i) for i in np.linspace(0, 1, 12)]

	levels = np.concatenate([[0, .0001], np.arange(1, 10), [50, 100]])
	cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

	zm = ma.masked_where(np.isnan(z), z)

	if annot:		# print values on grid
		offset = (1.41 / (len(x) - 1)) / 2
		for i, x_ in enumerate(x):
			for j, y_ in enumerate(y):
				plot_ax.text(x_ + offset, y_ + offset, '%.3f' % z[j, i],
						 horizontalalignment='center',
						 verticalalignment='center',
						 color='salmon'
						 )

	def extend_domain(x, y):
		'''
		from pcolormesh() documentation:

		Ideally, the dimensions of X and Y should
		be one greater than those of C; if the dimensions are the same, then
		the last row and column of C will be ignored.
		'''

		d = x[1] - x[0]

		x = np.append(x, x[-1] + d)
		y = np.append(y, y[-1] + d)
		return x, y


	x, y = extend_domain(x, y)

	plot_ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, clip_on=False)
	colorbar.ColorbarBase(cbar_ax, norm=norm, cmap=cmap, ticks=levels)

	return cmap


def make_PRF_plot(filtration, out_filename, PRF_res=50, params=None, in_filename=None, annot_hm=False):
	print "plotting PRF..."

	fig = pyplot.figure(figsize=(10, 6), tight_layout=True, dpi=300)
	fname_ax = 		pyplot.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	# epsilon_ax = 	pyplot.subplot2grid((6, 10), (1, 0), rowspan=1, colspan=3)
	params_ax = 	pyplot.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		pyplot.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		pyplot.subplot2grid((6, 10), (0, 9), rowspan=6)


	if isinstance(filtration, Filtration):
		func = filtration.get_PRF(PRF_res)
		in_filename = filtration.filename
		params = filtration.params
	else:
		func = filtration


	x, y, z, max_lim = func

	if len(x.shape) == 2: 			# meshgrid format
		x, y = x[0], y[:, 0]		# reduce to arange format


	plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=annot_hm)
	add_filename_table(fname_ax, in_filename)
	add_filt_params_table(params_ax, params)


	fig.savefig(out_filename)
	pyplot.close(fig)


from matplotlib import collections

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

	# ax.text(.9, .9, '(a)',
	# 		horizontalalignment='center',
	# 		transform=ax.transAxes)


	plt.savefig(out_filename)
	plt.close(fig)


def plot_PD_pub(filtration, out_filename):
	fig = pyplot.figure(figsize=(6, 6), dpi=500)
	ax = fig.add_subplot(111)
	add_persistence_plot(ax, filtration)
	pyplot.savefig(out_filename)
	pyplot.close(fig)
