import os
import sys
import time
import pickle
import subprocess

import numpy as np
import itertools

import BuildFiltration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_saved_filtration():
	caller_dir = os.getcwd()
	os.chdir(SCRIPT_DIR)
	filtration = pickle.load(open('temp_data/filtration.p'))
	os.chdir(caller_dir)
	return filtration



import subprocess
from config import find_landmarks_c_compile_str
class Filtration:

	def __init__(self, sig, params, filename=None, start=0):
		caller_dir = os.getcwd()

		if isinstance(sig, basestring):			# is filename
			self.sig = np.loadtxt(sig)
			self.filename = caller_dir + '/' + sig
		else:									# is array
			self.sig = sig
			self.filename = filename

		os.chdir(SCRIPT_DIR)

		self.params = params

		arr = self._build(params, start=0)

		self.witness_coords = arr[0]
		self.landmark_coords = arr[1]
		self.complexes = self._unpack_complexes(arr[2])
		self.epsilons = arr[3]

		self.ambient_dim = len(self.witness_coords[0])
		self.num_div = len(self.complexes)

		self.intervals = None
		self.PD_data = None
		self.PRF = None

		pickle.dump(self, open('temp_data/filtration.p', 'wb'))

		os.chdir(caller_dir)


	# private #
	def _build(self, params, start=0):
		print "building filtration..."
		start_time = time.time()
		np.savetxt('temp_data/worm_data.txt', self.sig)

		try:
			filtration = BuildFiltration.build_filtration('temp_data/worm_data.txt', params)
		except OSError:
			print "WARNING: invalid PH/find_landmarks binary. Recompiling..."
			print 'If problem persists, you will need to manually compile PH/find_landmarks.c. See config.py for default GCC commands.'

			if sys.platform == "linux" or sys.platform == "linux2":
				compile_str = find_landmarks_c_compile_str['linux']
			elif sys.platform == 'darwin':
				compile_str = find_landmarks_c_compile_str['macOS']
			else:
				print 'Sorry, PHETS requires linux or macOS.'
				sys.exit()

			subprocess.call(compile_str, shell=True)
			print "find_landmarks recompilation complete. Please repeat your test."
			print 'If problem persists, you will need to manually compile PH/find_landmarks.c. See config.py for default GCC commands.'

			sys.exit()


		witness_coords = filtration[1][1]
		landmark_coords = filtration[1][0]
		abstract_filtration = sorted(list(filtration[0]))
		epsilons = filtration[2]		# add to build_filtration return

		print("build_and_save_filtration() time elapsed: %d seconds \n" % (time.time() - start_time))
		return [witness_coords, landmark_coords, abstract_filtration, epsilons]

	def _unpack_complexes(self, filt_ID_list):

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
					expanded_row.extend(expanded_set)		# flatten
					# expanded_row.append(expanded_set)		# group by parent
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


				with open("output/run_info/num_triangles.txt","wb") as f:
					f.write("Number of triangles: "+str(len(triangles)))

		filt_ID_array = group_by_birth_time(filt_ID_list)	# 1d list -> 2d array
		expand_to_2simplexes(filt_ID_array)
		return filt_ID_array

	def _get_intervals(self):
		if self.intervals is not None:
			return

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

		def call_perseus():

			os.chdir('perseus')

			for f in os.listdir('.'):
				if f.startswith('perseus_out'):
					os.remove(f)

			if sys.platform == "linux" or sys.platform == "linux2":
				subprocess.call("./perseusLin nmfsimtop perseus_in.txt perseus_out", shell=True)

			elif sys.platform == "darwin":  # macOS
				subprocess.call("./perseusMac nmfsimtop perseus_in.txt perseus_out", shell=True)

			else:  # Windows
				subprocess.call("perseusWin.exe nmfsimtop perseus_in.txt perseus_out", shell=True)

			os.chdir('..')

		def load_perseus_out_file():
			self.intervals = np.loadtxt('perseus/perseus_out_1.txt', ndmin=1)
			if len(self.intervals) == 0:
				print 'WARNING: no homology for', self.filename
				self.intervals = 'empty'

		caller_dir = os.getcwd()
		os.chdir(SCRIPT_DIR)
		build_perseus_in_file(self.complexes)
		call_perseus()
		load_perseus_out_file()
		os.chdir(caller_dir)

	def _build_PD_data(self):
		""" formats perseus output """
		class PDData:
			def __init__(self, mortal, immortal, lim):
				self.mortal = mortal
				self.immortal = immortal
				self.lim = lim


		def get_multiplicity(birth_e, death_e):
			try:
				count = np.zeros_like(birth_e)
			except TypeError:  # only one interval point
				count = [0]
			if not death_e:
				death_e = [-1 for e in birth_e]
			for i, pt in enumerate(zip(birth_e, death_e)):
				for scanner_pt in zip(birth_e, death_e):
					if pt == scanner_pt:
						count[i] += 1
			return count

		if self.PD_data:
			return

		if isinstance(self.intervals, basestring):
			if self.intervals == 'empty':
				self.PD_data = 'empty'
				return

		epsilons = self.epsilons
		lim = np.max(epsilons)

		birth_e_mor = []
		death_e_mor = []

		birth_e_imm = []

		try:
			birth_t, death_t = self.intervals[:, 0], self.intervals[:, 1]
		except IndexError:
			print('WARNING: no homology for', self.filename)
			self.PD_data = 'empty'
			return

		for interval in zip(birth_t, death_t):
			if interval[1] == -1:	# immortal
				birth_e_imm.append(epsilons[int(interval[0] - 1)])

			else:
				birth_e_mor.append(epsilons[int(interval[0] - 1)])
				death_e_mor.append(epsilons[int(interval[1] - 1)])

		count_mor = get_multiplicity(birth_e_mor, birth_e_mor)
		mortal = np.asarray([birth_e_mor, death_e_mor, count_mor]).T
		mortal = np.vstack({tuple(row) for row in mortal}).T  # toss duplicates

		count_imm = get_multiplicity(birth_e_imm, None)
		immortal = np.asarray([birth_e_imm, count_imm]).T
		if len(immortal) != 0:
			immortal = np.vstack({tuple(row) for row in immortal}).T # toss duplicates


		data = PDData(mortal, immortal, lim)

		self.PD_data = data

	def _build_PRF(self, num_div):

		if self.PRF:
			return

		if self.PD_data == 'empty':
			print
			return [None, None, np.zeros([num_div, num_div]), None]

		x, y, z = self.get_PD_data().mortal
		max_lim = self.PD_data.lim
		min_lim = 0

		x_ = y_ = np.linspace(min_lim, max_lim, num_div)
		xx, yy = np.meshgrid(x_, y_)

		pts = zip(x, y, z)
		grid_pts = zip(np.nditer(xx), np.nditer(yy))
		grid_vals = np.zeros(len(grid_pts))
		for i, grid_pt in enumerate(grid_pts):
			if grid_pt[0] <= grid_pt[1]:
				for pt in pts:
					if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
						grid_vals[i] += pt[2]
			else:
				grid_vals[i] = np.nan
		grid_vals = np.reshape(grid_vals, xx.shape)

		self.PRF = [xx, yy, grid_vals, max_lim]


	# public #
	def get_complexes_mpl(self):

		# old version, for simplexes are not flattened #
		# def IDs_to_coords(ID_array):
		# 	"""Replaces each landmark_ID with corresponding coordinates"""
		# 	for row in ID_array:
		# 		for parent_simplex in row:
		# 			new_parent_simplex = []
		# 			for child in parent_simplex:
		# 				new_parent_simplex.append(list(child))
		# 			for child in new_parent_simplex:
		# 				new_child = []
		# 				for landmark_ID in child:
		# 					landmark_coords = self.landmark_coords[landmark_ID]
		# 					new_child.append(landmark_coords)
		# 				child[:] = new_child
		# 			parent_simplex[:] = new_parent_simplex

		def IDs_to_coords(ID_array):
			"""Replaces each landmark_ID with corresponding coordinates"""
			for row in ID_array:
				new_row = []
				for simplex in row:
					simplex_coords = []
					for landmark_ID in simplex:
						landmark_coords = self.landmark_coords[landmark_ID]
						simplex_coords.append(landmark_coords)
					new_row.append(simplex_coords)
				row[:] = new_row



		def flatten_rows(ID_array):
			for row in ID_array:
				new_row = []
				for parent in row:
					for child in parent:
						new_row.append(child)
				row[:] = new_row

		data = self.complexes
		IDs_to_coords(data)
		# flatten_rows(data)		# if grouped by parent simplex
		return data

	def get_complexes_mayavi(self):

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

		return separate_by_k(self.complexes)

	def get_PD_data(self):
		self._get_intervals()	# calls perseus, sets self.intervals
		self._build_PD_data()	# sets self.PD_data, returns PD_data
		return self.PD_data

	def get_PRF(self, num_div):
		self._get_intervals()
		self._build_PD_data()
		self._build_PRF(num_div)
		return self.PRF

