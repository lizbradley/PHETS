import os
import BuildFiltration
import numpy as np
import time

class Filtration:

	def __init__(self, in_filename, params, start=0):
		if in_filename == 'load':
			print "WARNING: in_filename='load'. Reusing saved filtration."
			arr = np.load('temp_data/filtration.npy')

		else:
			arr = self.build_and_save(in_filename, params, start=0)

		self.witness_coords = arr[0]
		self.landmark_coords = arr[1]
		self.abstract_filtration = self.unpack(arr[2])



	# private #
	def build_and_save(self, in_file_name, params, start=0):
		print "building filtration..."
		start_time = time.time()

		abspath = os.path.abspath(__file__)
		dname = os.path.dirname(abspath)
		os.chdir(dname)

		# sliding window #
		lines = open(in_file_name).readlines()
		start_idx = int(len(lines) * start)
		open('temp_data/worm_data.txt', 'w').writelines(lines[start_idx:])

		filtration = BuildFiltration.build_filtration('temp_data/worm_data.txt', params)
		witness_coords = filtration[1][1]
		landmark_coords = filtration[1][0]
		abstract_filtration = sorted(list(filtration[0]))

		np.save('temp_data/filtration.npy', [witness_coords, landmark_coords, abstract_filtration])
		print("build_and_save_filtration() time elapsed: %d seconds \n" % (time.time() - start_time))
		return [witness_coords, landmark_coords, abstract_filtration]

	def unpack(self, filt_ID_list):

		def group_by_birth_time(ID_list):
			"""Reformats 1D list of SimplexBirth objects into 2D array of
			landmark_set lists, where 2nd index is  birth time (? see below)"""

			ID_array = []  # list of complex_at_t lists
			complex_at_t = []  # list of simplices with same birth_time
			i = 0
			time = 0
			list_length = len(ID_list)
			while i < list_length:
				birth_time = ID_list[i].birth_time
				if birth_time == time:
					complex_at_t.append(ID_list[i].landmark_set)
					if i == list_length - 1:
						ID_array.append(complex_at_t)
					i += 1
				else:
					ID_array.append(complex_at_t)
					complex_at_t = []
					time += 1
			return ID_array

		def expand_to_2simplexes(ID_array):
			"""for each k-simplex in filtration array, if k > 2, replaces with the
			component 2-simplexes(i.e. all length-3 subsets of landmark_ID_set) """
			for row in ID_array:
				expanded_row = []
				for landmark_ID_set in row:
					if len(landmark_ID_set) > 3:
						expanded_set = list(itertools.combinations(landmark_ID_set, 3))
					else:
						expanded_set = [list(landmark_ID_set)]
					expanded_row.extend(expanded_set)
				row[:] = expanded_row


			def count_triangles():
				print 'counting triangles...'
				num_tris=0
				count =0
				triangles=[]
				for i in ID_array:
					for j in i:
						if len(j)==3:
							tri=set(j)
							if tri not in triangles:
								triangles.append(tri)


				with open("PersistentHomology/output/num_triangles.txt","wb") as f:
					f.write("Number of triangles: "+str(len(triangles)))

		filt_ID_array = group_by_birth_time(filt_ID_list)	# 1d list -> 2d array
		expand_to_2simplexes(filt_ID_array)
		return filt_ID_array
	# end private #



	# public #
	def get_data_for_mpl(self):
		def IDs_to_coords(ID_array):
			"""Replaces each landmark_ID with corresponding coordinates"""
			for row in ID_array:
				for parent_simplex in row:
					new_parent_simplex = []
					for child in parent_simplex:
						new_parent_simplex.append(list(child))
					for child in new_parent_simplex:
						new_child = []
						for landmark_ID in child:
							landmark_coords = self.landmark_coords[landmark_ID]
							new_child.append(landmark_coords)
						child[:] = new_child
					parent_simplex[:] = new_parent_simplex

		def flatten_rows(ID_array):
			for row in ID_array:
				new_row = []
				for parent in row:
					for child in parent:
						new_row.append(child)
				row[:] = new_row

		data = self.abstract_filtration
		IDs_to_coords(data)
		flatten_rows(data)
		return data


	def get_data_for_mayavi(self):
		def separate_by_k(array):
			lines = []
			triangles = []
			for row in array:
				lines_row = []
				triangles_row = []
				for simplex in row:
					if len(simplex) == 2:
						lines_row.append(simplex)
					else:  # if len(simplex) == 3:
						triangles_row.append(simplex)
				triangles.append(triangles_row)
				lines.append(lines_row)
			return [lines, triangles]

		return separate_by_k(self.abstract_filtration)


	def get_intervals(self):
		pass

	def get_PRF(self):
		pass

	# end public #

###########################################################################################
#	from PersitencePlotter.py #
###########################################################################################



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


# duplicates PRF.get_homology()
def add_persistence_plot(subplot):
	print 'plotting persistence diagram...'
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


	subplot.plot([min_lim, max_lim], [min_lim, max_lim], color='k')  # diagonal line

	# plot immortal holes #
	immortal_holes = [epsilons[int(birth_t[i]) - 1] for i, death_time in enumerate(death_t) if death_time == -1]
	count = np.zeros(len(immortal_holes))
	for i, pt in enumerate(immortal_holes):
		for scanner_pt in immortal_holes:
			if pt == scanner_pt:
				count[i] += 1

	# normal #
	min_size = 0
	t_ms_scale = 50
	p_ms_scale = 30
	color = 'C0'

	# BIG for paper #
	# min_size = 300
	# t_ms_scale = 150
	# p_ms_scale = 60
	# color = 'red'

	x, y = immortal_holes, [max_lim for i in immortal_holes]
	subplot.scatter(x, y, marker='^', s=(count * t_ms_scale) + min_size, c=color, clip_on=False)
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

	subplot.scatter(birth_e, death_e, s=(count * p_ms_scale) + min_size, clip_on=False, c=color)
	# end plot doomed holes #



	# add legend #
	mark_t_1 = subplot.scatter([], [], marker='^', s=t_ms_scale, c=color)
	mark_t_3 = subplot.scatter([], [], marker='^', s=t_ms_scale * 3, c=color)
	mark_t_5 = subplot.scatter([], [], marker='^', s=t_ms_scale * 5, c=color)

	mark_p_1 = subplot.scatter([], [], s=p_ms_scale, c=color)
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
		# edgecolor='k'
	)


# end add legend #

###########################################################################################
#	from FiltrationPlotter.py #
###########################################################################################


###########################################################################################
#	from PRF.py #
###########################################################################################

# duplicates PersistencePlotter.add_persistence_plot()
def get_homology(filt_list):
	""" calls perseus, creating perseus_out_*.txt
		TODO: move to PersistentHomology and replace equivalent code there
	"""

	def build_perseus_in_file(filt_array):
		print 'building perseus_in.txt...'
		out_file = open('perseus/perseus_in.txt', 'a')
		out_file.truncate(0)
		out_file.write('1\n')
		for idx, row in enumerate(filt_array):
			for simplex in row:
				#   format for perseus...
				line_str = str(len(simplex) - 1) + ' ' + ' '.join(
					str(ID) for ID in simplex) + ' ' + str(idx + 1) + '\n'
				out_file.write(line_str)
		out_file.close()

	filt_array = group_by_birth_time(filt_list)
	expand_to_2simplexes(filt_array)
	filt_array = np.asarray(filt_array)
	build_perseus_in_file(filt_array)

	print 'calling perseus...'
	os.chdir('perseus')

	if platform == "linux" or platform == "linux2":
		subprocess.call("./perseusLin nmfsimtop perseus_in.txt perseus_out", shell=True)

	elif platform == "darwin":  # macOS
		subprocess.call("./perseusMac nmfsimtop perseus_in.txt perseus_out", shell=True)

	else:   # Windows
		subprocess.call("perseusWin.exe nmfsimtop perseus_in.txt perseus_out", shell=True)

	os.chdir('..')
	os.chdir('..')



def get_interval_data(filename):
	""" formats perseus output """
	# NOTE: should be merged back into PersistencePlotter
	try:
		birth_t, death_t = np.loadtxt('PersistentHomology/perseus/perseus_out_1.txt', unpack=True, ndmin=1)
	except ValueError:
		print 'WARNING: no homology for', filename
		return None

	epsilons = np.loadtxt('PersistentHomology/temp_data/epsilons.txt')
	lim = np.max(epsilons)

	birth_e = []
	death_e = []

	timess = np.vstack([birth_t, death_t]).T
	for times in timess:
		if times[1] != - 1:
			birth_e.append(epsilons[int(times[0])])
			death_e.append(epsilons[int(times[1])])

	immortal_holes = []
	for i, death_time in np.ndenumerate(death_t):    # place immortal holes at [birth time, time lim]
		if death_time == -1:
			immortal_holes.append([epsilons[int(birth_t[i])], lim * .95])
	immortal_holes = np.array(immortal_holes)

	if len(immortal_holes):
		birth_e.extend(immortal_holes[:,0])
		death_e.extend(immortal_holes[:,1])

	try:
		count = np.zeros(len(birth_t))
	except TypeError:		# only one interval point
		count = [0]
	for i, pt in enumerate(zip(birth_e, death_e)):
		for scanner_pt in zip(birth_e, death_e):
			if pt == scanner_pt:
				count[i] += 1

	points = np.asarray([birth_e, death_e, count]).T
	points = np.vstack({tuple(row) for row in points})  # toss duplicates

	x, y, z = points[:,0], points[:,1], points[:,2]

	return x, y, z, lim


