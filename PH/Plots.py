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

from Data import Filtration

f=open("output/run_info/group_by_birth_time_memory.txt","wb")
f2=open("output/run_info/expand_to_2simplexes_memory.txt","wb")
f3=open("output/run_info/build_perseus_in_file_memory.txt","wb")
f4=open("output/run_info/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)

def add_filename_table(ax, filenames):
	ax.axis('off')
	title_table = ax.table(
		cellText=[
			[filenames.split('/')[-1]],   # remove leading "datasets/"
		],
		bbox=[0, 0, 1, 1],
		cellLoc='center'
	)
	# title_table.auto_set_font_size(False)
	# title_table.auto_set_font_size(8)


def add_params_table(subplot, filt_params):
	subplot.axis('off')
	subplot.set_xlim([0,1])
	subplot.set_ylim([0,1])


	display_params = (
		"max_filtration_param",
		"min_filtration_param",
		'num_divisions',
		"start",
		"worm_length",
		"ds_rate",
		"landmark_selector",
		"d_orientation_amplify",
		"d_use_hamiltonian",
		"d_cov",
		"simplex_cutoff",
		"use_cliques",
		"use_twr",
		"m2_d",
		"straight_VB",
		"dimension_cutoff",
	)
	param_data = np.array([[key, filt_params[key]] for key in display_params])
	param_table = subplot.table(
		cellText=param_data,
		colWidths=[1.5, .5],
		bbox=[0, 0, 1, 1],  # x0, y0, width, height
	)
	param_table.auto_set_font_size(False)
	param_table.set_fontsize(6)




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
	add_params_table(params_ax, filt.params)
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


def make_PRF_plot(filt, out_filename, PRF_res=50, params=None, in_filename=None):
	print "\nplotting PRF... \n"

	if isinstance(filt, Filtration):
		func = filt.get_PRF(PRF_res)
		in_filename = filt.filename
		params = filt.params
	else:
		func = filt

	fig = pyplot.figure(figsize=(10, 6), tight_layout=True, dpi=300)
	fname_ax = 		pyplot.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	# epsilon_ax = 	pyplot.subplot2grid((6, 10), (1, 0), rowspan=1, colspan=3)
	params_ax = 	pyplot.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		pyplot.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		pyplot.subplot2grid((6, 10), (0, 9), rowspan=6)


	x, y, z, max_lim = func
	# z = np.log10(z + 1)

	if not max_lim:
		x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
		z = np.zeros((10, 10))
		max_lim = 1

	plot_ax.set_xlim([0, max_lim])
	plot_ax.set_ylim([0, max_lim])
	plot_ax.set_aspect('equal')

	cm = plot_ax.contourf(x, y, z)
	fig.colorbar(cm, cax=cbar_ax)

	add_filename_table(fname_ax, in_filename)
	add_params_table(params_ax, params)


	fig.savefig(out_filename)




if __name__ == '__main__':
	make_PD('filt_test.txt')