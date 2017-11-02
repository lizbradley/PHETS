import matplotlib.colors as colors
import matplotlib.pyplot as pyplot
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

pyplot.ioff()

import numpy as np

from Data import Filtration
from TitleBox import filename_table, filt_params_table

# from Utilities import mem_profile
import os
print os.getcwd()
# f=open("output/run_info/group_by_birth_time_memory.txt","wb")
# f2=open("output/run_info/expand_to_2simplexes_memory.txt","wb")
# f3=open("output/run_info/build_perseus_in_file_memory.txt","wb")
# f4=open("output/run_info/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)



def PD_ax(ax, filtration):
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

	# BIG for IDA IDA2017 #
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

	fig = pyplot.figure(figsize=(10, 6), tight_layout=True, dpi=700)

	fname_ax = 		pyplot.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	# epsilon_ax = 	pyplot.subplot2grid((6, 10), (1, 0), rowspan=1, colspan=3)
	params_ax = 	pyplot.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		pyplot.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)

	PD_ax(plot_ax, filt)
	filename_table(fname_ax, filt.filename)
	filt_params_table(params_ax, filt.params)


	pyplot.savefig(out_filename)
	pyplot.close(fig)


import matplotlib
import matplotlib.colorbar as colorbar
import numpy.ma as ma


def plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=False):
	plot_ax.set_aspect('equal')

	viridis = matplotlib.cm.get_cmap('viridis')
	colors = [viridis(i) for i in np.linspace(0, 1, 13)]

	levels = np.concatenate([[0, .0001], np.arange(1, 10), [50, 100]])
	cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend='max')

	zm = ma.masked_where(np.isnan(z), z)

	def annotate():
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


	if x is not None and y is not None:
		x, y = extend_domain(x, y)
		plot_ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, clip_on=False)
		if annot: annotate()
	elif x is None and y is None:
		plot_ax.pcolormesh(zm, cmap=cmap, norm=norm, clip_on=False)
	else:
		print 'ERROR: plot_heatmap: x and y must both be None or array-like'
		sys.exit()

	colorbar.ColorbarBase(cbar_ax, norm=norm, cmap=cmap, ticks=levels, extend='max')

	return cmap


def PRF_ax(filtration, ax, cbar_ax=None, annot_hm=False):


	if cbar_ax is None:
		divider = make_axes_locatable(ax)
		cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

	if isinstance(filtration, Filtration):
		z = filtration.get_PRF()
		x = y = filtration.epsilons
		plot_heatmap(ax, cbar_ax, x, y, z, annot_hm)
	else:   # 2d array
		z = filtration
		plot_heatmap(ax, cbar_ax, None, None, z, annot_hm)




def make_PRF_plot(filtration, out_filename, params=None, in_filename=None,
				  annot_hm=False):
	print "plotting PRF..."

	fig = pyplot.figure(figsize=(10, 6), tight_layout=True, dpi=300)
	fname_ax = 		pyplot.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	# epsilon_ax = 	pyplot.subplot2grid((6, 10), (1, 0), rowspan=1, colspan=3)
	params_ax = 	pyplot.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		pyplot.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		pyplot.subplot2grid((6, 10), (0, 9), rowspan=6)

	######## from here ##########

	if isinstance(filtration, Filtration):
		func = filtration.get_PRF()
		in_filename = filtration.filename
		params = filtration.params
	else:
		func = filtration

	x, y, z, max_lim = func

	if len(x.shape) == 2: 			# meshgrid format
		x, y = x[0], y[:, 0]		# reduce to arange format


	plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=annot_hm)

	####### to here ###########
	# should eventually be replaced by PRF_ax

	filename_table(fname_ax, in_filename)
	filt_params_table(params_ax, params)


	fig.savefig(out_filename)
	pyplot.close(fig)


