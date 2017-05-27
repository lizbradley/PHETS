import subprocess
import os
import matplotlib.pyplot as pyplot
import itertools
import numpy as np
from os import system, chdir
from memory_profiler import profile
from sys import platform

from matplotlib.ticker import FormatStrFormatter


# from Utilities import mem_profile
from config import MEMORY_PROFILE_ON


f=open("output/run_info/group_by_birth_time_memory.txt","wb")
f2=open("output/run_info/expand_to_2simplexes_memory.txt","wb")
f3=open("output/run_info/build_perseus_in_file_memory.txt","wb")
f4=open("output/run_info/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)

def add_title(subplot, title_block_info):
	in_file_name = title_block_info[0]
	out_file_name = title_block_info[1]
	parameter_set = title_block_info[2]

	# subplot.axis('tight')
	subplot.axis('off')
	subplot.set_xlim([0,1])
	subplot.set_ylim([0,1])

	param_data = np.array(
		[[key, parameter_set[key]] for key in parameter_set.keys()])
	param_table = subplot.table(
		cellText=param_data,
		colWidths=[1.5, .5],
		bbox=[0, 0, 1, .8],  # x0, y0, width, height
	)
	param_table.auto_set_font_size(False)
	param_table.set_fontsize(4.5)

	title_table = subplot.table(
		cellText = [[in_file_name.split('/')[-1]],   # remove leading "datasets/"
					[out_file_name.split('/')[-1]]],
		bbox=[0, .9, 1, .1],
	)
	title_table.auto_set_font_size(False)
	title_table.auto_set_font_size(8)

def add_persistence_plot(subplot, filtration):

	subplot.set_aspect('equal')
	min_lim = 0
	max_lim = np.max(filtration.epsilons)
	subplot.set_xlim(min_lim, max_lim)
	subplot.set_ylim(min_lim, max_lim)

	# subplot.set_xlabel('birth time')
	# subplot.set_ylabel('death time')


	subplot.plot([min_lim, max_lim], [min_lim, max_lim], color='k') # diagonal line


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

	immortal_holes, birth_e, death_ = filtration.PD_data()
	x, y = immortal_holes, [max_lim for i in immortal_holes]
	subplot.scatter(x, y, marker='^', s=(count * t_ms_scale) + min_size, c=color, clip_on=False)
	# end plot immortal holes#


	subplot.scatter(birth_e, death_e, s=(count * p_ms_scale) + min_size, clip_on=False, c=color)
	# end plot doomed holes #



	# add legend #
	mark_t_1 = subplot.scatter([], [], marker='^', s=t_ms_scale,	 c=color)
	mark_t_3 = subplot.scatter([], [], marker='^', s=t_ms_scale * 3, c=color)
	mark_t_5 = subplot.scatter([], [], marker='^', s=t_ms_scale * 5, c=color)

	mark_p_1 = subplot.scatter([], [], s=p_ms_scale,	 c=color)
	mark_p_3 = subplot.scatter([], [], s=p_ms_scale * 3, c=color)
	mark_p_5 = subplot.scatter([], [], s=p_ms_scale * 5, c=color)

	marks = (mark_t_1, mark_t_3, mark_t_5, mark_p_1, mark_p_3, mark_p_5)
	labels = ('', '', '', '1', '3', '5')

	subplot.legend(
		marks, labels, loc='lower right', ncol=2, markerscale=1,
		borderpad=1,
		labelspacing=1,
		framealpha=1,
		columnspacing=0,
		borderaxespad=3
		#edgecolor='k'
	)
	# end add legend #


# @profile(stream=f4)
def make_PD(filtration, out_filename):

	fig = pyplot.figure(figsize=(8,6), tight_layout=True, dpi=300)
	title_plot = pyplot.subplot2grid((3, 4), (0, 0), rowspan=3)
	ax = pyplot.subplot2grid((3, 4), (0, 1), rowspan=3, colspan=3)

	title_info = [filtration.filename, out_filename, filtration.params]

	add_persistence_plot(ax, )
	add_title(title_plot, title_info)

	# IDA paper figures #
	# title_block.tick_params(labelsize=23)
	# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	xlims = ax.get_xlim()
	ax.set_xticks([0, round(xlims[1]/2., 4), xlims[1]])
	ylims = ax.get_ylim()
	ax.set_yticks([round(ylims[1]/2., 4), ylims[1]])
	ax.tick_params(labelsize=23)

	ax.grid(which=u'both', zorder=0)
	ax.minorticks_on()
	title_plot.axis('off')
	
	pyplot.savefig(out_filename)
	pyplot.clf()


if __name__ == '__main__':
	make_PD('filt_test.txt')