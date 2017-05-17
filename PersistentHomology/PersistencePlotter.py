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


f=open("PersistentHomology/output/group_by_birth_time_memory.txt","wb")
f2=open("PersistentHomology/output/expand_to_2simplexes_memory.txt","wb")
f3=open("PersistentHomology/output/build_perseus_in_file_memory.txt","wb")
f4=open("PersistentHomology/output/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)
def group_by_birth_time(complex_ID_list):
	"""Reformats 1D list of SimplexBirth objects into 2D array of
	landmark_set lists, where 2nd index is  birth time (? see below)"""

	# TODO: ensure that if a time t has no births, the row t is empty/skipped

	complex_ID_array = []  # list of complex_at_t lists
	complex_at_t = []  # list of simplices with same birth_time
	i = 0
	time = 0
	list_length = len(complex_ID_list)
	while i < list_length:
		birth_time = complex_ID_list[i].birth_time
		if birth_time == time:
			complex_at_t.append(complex_ID_list[i].landmark_set)
			if i == list_length - 1:		# edge case
				complex_ID_array.append(complex_at_t)
			i += 1
		else:
			complex_ID_array.append(complex_at_t)
			complex_at_t = []
			time += 1
	return complex_ID_array


# @profile(stream=f2)
def expand_to_2simplexes(filt_array):
	"""for each k-simplex in filtration array, if k > 2, replaces with the
	component 2-simplexes(i.e. all length-3 subsets of landmark_ID_set) """
	for row in filt_array:
		expanded_row = []
		for landmark_ID_set in row:
			expanded_set = list(itertools.combinations(landmark_ID_set, 3)) \
				if len(landmark_ID_set) > 3 else [list(landmark_ID_set)]
			expanded_row.extend(expanded_set)
		row[:] = expanded_row
	num_tris=0
	
	
	count =0
	triangles=[]
	for i in filt_array:
		for j in i:
			if len(j)==3:
				tri=set(j)
				if tri not in triangles:
					triangles.append(tri)
				
	
	
	with open("PersistentHomology/output/num_triangles.txt","wb") as f:
		f.write("Number of triangles: "+str(len(triangles)))



# @profile(stream=f3)
def build_perseus_in_file(filt_array):
	print 'building perseus_in.txt...'
	out_file = open('PersistentHomology/perseus/perseus_in.txt', 'a')
	out_file.truncate(0)
	out_file.write('1\n')
	for idx, row in enumerate(filt_array):
		for simplex in row:
			#   format for perseus...
			line_str = str(len(simplex) - 1) + ' ' + ' '.join(
				str(ID) for ID in simplex) + ' ' + str(idx + 1) + '\n'
			out_file.write(line_str)
	out_file.close()


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

def add_persistence_plot(subplot):
	print 'preparing persistence diagram...'
	birth_t, death_t = np.loadtxt('PersistentHomology/perseus/perseus_out_1.txt', unpack=True)

	epsilons = np.loadtxt('PersistentHomology/temp_data/epsilons.txt')
	max_lim = np.max(epsilons)
	# min_lim = np.min(epsilons)
	min_lim = 0

	subplot.set_aspect('equal')



	subplot.set_xlim(min_lim, max_lim)
	subplot.set_ylim(min_lim, max_lim)
	 
	# subplot.set_xlabel('birth time')
	# subplot.set_ylabel('death time')


	subplot.plot([min_lim, max_lim], [min_lim, max_lim], color='k') # diagonal line

	# plot immortal holes #
	immortal_holes = [epsilons[int(birth_t[i]) - 1] for i, death_time in enumerate(death_t) if death_time == -1]
	count = np.zeros(len(immortal_holes))
	for i, pt in enumerate(immortal_holes):
		for scanner_pt in immortal_holes:
			if pt == scanner_pt:
				count[i] += 1

	t_ms_scale = 50
	p_ms_scale = 30

	min_size = 300

	# t_ms_scale = 150
	# p_ms_scale = 60

	x, y = immortal_holes, [max_lim for i in immortal_holes]
	subplot.scatter(x, y, marker='^', s=(count * t_ms_scale) + min_size, c='red', clip_on=False)
	# end plot immortal holes#



	# plot doomed holes #
	birth_e, death_e = [], []
	for times in zip(birth_t, death_t):
		if times[1] != - 1:
			birth_e.append(epsilons[int(times[0] - 1)])
			death_e.append(epsilons[int(times[1] - 1)])

	count = np.zeros(len(birth_t))
	for i, pt in enumerate(zip(birth_t, death_t)):
		for scanner_pt in zip(birth_t, death_t):
			if pt == scanner_pt:
				count[i] += 1

	subplot.scatter(birth_e, death_e, s=(count * p_ms_scale) + min_size, clip_on=False, c='red')
	# end plot doomed holes #



	# add legend #
	mark_t_1 = subplot.scatter([], [], marker='^', s=t_ms_scale, c='red')
	mark_t_3 = subplot.scatter([], [], marker='^', s=t_ms_scale * 3, c='red')
	mark_t_5 = subplot.scatter([], [], marker='^', s=t_ms_scale * 5, c='red')

	mark_p_1 = subplot.scatter([], [], s=p_ms_scale, c='red')
	mark_p_3 = subplot.scatter([], [], s=p_ms_scale * 3, c='red')
	mark_p_5 = subplot.scatter([], [], s=p_ms_scale * 5, c='red')

	marks = (mark_t_1, mark_t_3, mark_t_5, mark_p_1, mark_p_3, mark_p_5)
	labels = ('', '', '', '1', '3', '5')

	# subplot.legend(
	# 	marks, labels, loc='lower right', ncol=2, markerscale=1,
	# 	borderpad=1,
	# 	labelspacing=1,
	# 	framealpha=1,
	# 	columnspacing=0,
	# 	borderaxespad=3
	# 	#edgecolor='k'
	# )
	# end add legend #


# @profile(stream=f4)
def make_figure(title_block_info, out_file_name):
	print 'plotting persistence diagram...'
	filt_list = np.load('PersistentHomology/temp_data/complexes.npy')
	filt_array = group_by_birth_time(filt_list)
	expand_to_2simplexes(filt_array)
	filt_array = np.asarray(filt_array)


	build_perseus_in_file(filt_array)
	print 'calling perseus...'
	os.chdir('PersistentHomology/perseus')

	if platform == "linux" or platform == "linux2":
		subprocess.call("./perseusLin nmfsimtop perseus_in.txt perseus_out", shell=True)

	elif platform == "darwin":  # macOS
		subprocess.call("./perseusMac nmfsimtop perseus_in.txt perseus_out", shell=True)

	else:   # Windows
		subprocess.call("perseusWin.exe nmfsimtop perseus_in.txt perseus_out", shell=True)

	os.chdir('..')
	os.chdir('..')


	fig = pyplot.figure(figsize=(8,6), tight_layout=True, dpi=300)
	title_block = pyplot.subplot2grid((3, 4), (0, 0), rowspan=3)
	pers_plot = pyplot.subplot2grid((3, 4), (0, 1), rowspan=3, colspan=3)

	add_persistence_plot(pers_plot)

	# add_title(title_block, title_block_info)

	# paper figures
	# title_block.tick_params(labelsize=23)
	title_block.axis('off')

	ax = pers_plot
	xlims = ax.get_xlim()
	ax.set_xticks([0, round(xlims[1]/2., 4), xlims[1]])
	ylims = ax.get_ylim()
	ax.set_yticks([round(ylims[1]/2., 4), ylims[1]])
	pers_plot.tick_params(labelsize=23)

	# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	ax.grid(which=u'both', zorder=0)
	ax.minorticks_on()

	
	pyplot.savefig(out_file_name)
	pyplot.clf()


if __name__ == '__main__':
	make_figure('filt_test.txt')