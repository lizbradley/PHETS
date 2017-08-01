import os
import sys
import time
import pickle
import subprocess

import numpy as np
import itertools

import BuildFiltration
from Utilities import blockPrint, enablePrint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_saved_filtration():
	print 'loading saved filtration...'
	caller_dir = os.getcwd()
	os.chdir(SCRIPT_DIR)
	filtration = pickle.load(open('temp_data/filtration.p'))
	os.chdir(caller_dir)
	return filtration



import subprocess
from config import find_landmarks_c_compile_str


class PDData:
	def __init__(self, mortal, immortal, lim):
		self.mortal = mortal
		self.immortal = immortal
		self.lim = lim



class Filtration:

	def __init__(self, sig, params, filename='none', silent=False):
		caller_dir = os.getcwd()

		if isinstance(sig, basestring):			# is filename
			self.sig = np.loadtxt(sig)
			self.filename = caller_dir + '/' + sig
		else:									# is array
			self.sig = sig
			self.filename = filename

		os.chdir(SCRIPT_DIR)

		self.params = params

		arr = self._build(params, silent)

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
	def _build(self, params, silent):

		def compile_find_landmarks_c():
			if sys.platform == "linux" or sys.platform == "linux2":
				compile_str = find_landmarks_c_compile_str['linux']
			elif sys.platform == 'darwin':
				compile_str = find_landmarks_c_compile_str['macOS']
			else:
				print 'Sorry, PHETS requires linux or macOS.'
				sys.exit()
			subprocess.call(compile_str, shell=True)
			print "find_landmarks recompilation attempt complete. If successful (ignore warnings), please repeat your test."
			print 'If problem persists, you will need to manually compile PH/find_landmarks.c. See config.py for default GCC commands.'

			sys.exit()


		if not silent: print "building filtration..."

		if len(self.sig.shape) == 1:
			print "ERROR: Filtration input 'sig' is one dimensional"
			sys.exit()
		np.savetxt('temp_data/worm_data.txt', self.sig)
		start_time = time.time()

		try:
			if silent: blockPrint()
			filtration = BuildFiltration.build_filtration('temp_data/worm_data.txt', params, silent=silent)
			if silent: enablePrint()
		except OSError:
			print "WARNING: invalid PH/find_landmarks binary. Recompiling..."
			compile_find_landmarks_c()

		os.remove('temp_data/worm_data.txt')

		witness_coords = filtration[1][1]
		landmark_coords = filtration[1][0]
		abstract_filtration = sorted(list(filtration[0]))
		epsilons = filtration[2]		# add to build_filtration return

		if not silent: print("build_filtration() time elapsed: %d seconds \n" % (time.time() - start_time))
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


			# '''check for duplicate triangles in order of birth time'''
			# all_tris=set()
			# for row in ID_array:
			# 	unique_row=[]
			# 	for tri in row:
			# 		tr=set(tri)
			# 		if tr not in all_tris:
			# 			unique_row.append(tri)
			# 			all_tris.add(tr)
			# 	row[:]=unique_row


		filt_ID_array = group_by_birth_time(filt_ID_list)		# 1d list -> 2d array
		expand_to_2simplexes(filt_ID_array)
		# add _remove_duplicates() here IFF we want to process data before going in to perseus
		# might run faster if we don't give perseus a filtration where simplexes are reborn
		return filt_ID_array

	def _get_intervals(self, silent=False):

		def build_perseus_in_file(filt_array):
			if not silent: print 'building perseus_in.txt...'
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
				perseus_cmd = "./perseusLin nmfsimtop perseus_in.txt perseus_out"

			elif sys.platform == "darwin":  # macOS
				perseus_cmd = "./perseusMac nmfsimtop perseus_in.txt perseus_out"

			else:  # Windows
				perseus_cmd = "perseusWin.exe nmfsimtop perseus_in.txt perseus_out"

			if silent:
				p = subprocess.Popen(perseus_cmd, shell=True, stdout=subprocess.PIPE)
				out, err = p.communicate()

			else:
				p = subprocess.Popen(perseus_cmd, shell=True)
				p.communicate()		# wait

			os.chdir('..')

		def load_perseus_out_file():
			self.intervals = np.loadtxt('perseus/perseus_out_1.txt', ndmin=1)
			if len(self.intervals) == 0:
				print 'WARNING: no homology for this window!'
				self.intervals = 'empty'

		if self.intervals is not None:
			return

		build_perseus_in_file(self.complexes)
		call_perseus()
		load_perseus_out_file()


	def _build_PD_data(self):
		""" formats perseus output """

		def get_multiplicity(birth_e, death_e):
			try:
				count = np.zeros_like(birth_e)
			except TypeError:  # only one interval point
				count = [0]
			if death_e is None:
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

		def t_to_eps(t):
			return epsilons[int(t - 1)]

		if len(self.intervals.shape) == 1:		# one interval
			birth_t, death_t = self.intervals
			if death_t == -1:
				immortal = [t_to_eps(birth_t), 1]
				mortal = []
			else:
				immortal = []
				mortal = [t_to_eps(birth_t), t_to_eps(death_t), 1]

			data = PDData(mortal, immortal, lim)
			self.PD_data = data
			return

		birth_t, death_t = self.intervals[:, 0], self.intervals[:, 1]

		for interval in zip(birth_t, death_t):
			if interval[1] == -1:				# immortal
				birth_e_imm.append(t_to_eps(interval[0]))
			else:
				birth_e_mor.append(t_to_eps(interval[0]))
				death_e_mor.append(t_to_eps(interval[1]))

		count_mor = get_multiplicity(birth_e_mor, death_e_mor)
		mortal = np.asarray([birth_e_mor, death_e_mor, count_mor]).T
		if len(mortal):
			mortal = np.vstack({tuple(row) for row in mortal}).T  # toss duplicates

		count_imm = get_multiplicity(birth_e_imm, None)
		immortal = np.asarray([birth_e_imm, count_imm]).T
		if len(immortal):
			immortal = np.vstack({tuple(row) for row in immortal}).T # toss duplicates


		data = PDData(mortal, immortal, lim)

		self.PD_data = data


	def _build_PRF(self, num_div):

		if self.PRF:
			return

		if self.PD_data == 'empty':
			eps = np.asarray(self.epsilons)
			self.PRF = [eps, eps, np.zeros([num_div, num_div]), eps[-1]]
			return
		max_lim = self.PD_data.lim
		min_lim = 0

		x_ = y_ = np.linspace(min_lim, max_lim, num_div)
		xx, yy = np.meshgrid(x_, y_)

		try:
			x, y, z = self.get_PD_data().mortal
			try:
				pts = zip(x, y, z)
			except TypeError: 			# one interval
				pts = [[x, y, z]]
		except ValueError:				# no intervals
			pts = []

		try:
			x_imm, z_imm = self.get_PD_data().immortal
			try:
				pts_imm = zip(x_imm, z_imm)
			except TypeError:
				pts_imm = [[x_imm, z_imm]]
		except ValueError:
			pts_imm = []


		grid_pts = zip(np.nditer(xx), np.nditer(yy))
		grid_vals = np.zeros(len(grid_pts))
		for i, grid_pt in enumerate(grid_pts):
			if grid_pt[0] <= grid_pt[1]:
				for pt in pts:
					if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
						grid_vals[i] += pt[2]
				for pt in pts_imm:
					if pt[0] <= grid_pt[0]:
						grid_vals[i] += pt[1]
			else:
				grid_vals[i] = np.nan
		grid_vals = np.reshape(grid_vals, xx.shape)

		self.PRF = [xx, yy, grid_vals, max_lim]


	# public #
	def get_complex_plot_data(self, remove_dups=False):

		def IDs_to_coords(ID_array):
			"""Replaces each landmark_ID with corresponding coordinates"""
			coords_array = []
			for row in ID_array:
				new_row = []
				for simplex in row:
					simplex_coords = []
					for landmark_ID in simplex:
						landmark_coords = self.landmark_coords[landmark_ID]
						simplex_coords.append(landmark_coords)
					new_row.append(simplex_coords)
				coords_array.append(new_row)
			return np.asarray(coords_array)



		ID_array = self.complexes
		# if remove_dups: ID_array = remove_duplicates(ID_array)
		coords_array = IDs_to_coords(ID_array)
		return coords_array


	def get_PD_data(self):
		caller_dir = os.getcwd()
		os.chdir(SCRIPT_DIR)
		self._get_intervals()		# calls perseus, sets self.intervals
		self._build_PD_data()		# sets self.PD_data, returns PD_data
		os.chdir(caller_dir)
		return self.PD_data

	def get_PRF(self, num_div, silent=False):
		caller_dir = os.getcwd()
		os.chdir(SCRIPT_DIR)
		self._get_intervals(silent=silent)
		self._build_PD_data()
		self._build_PRF(num_div)
		os.chdir(caller_dir)
		return self.PRF

