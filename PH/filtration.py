import os, sys, time, cPickle, warnings, subprocess, itertools
import numpy as np

import build_filtration, plots, filtration_movie
from utilities import blockPrint, enablePrint
from config import find_landmarks_c_compile_str

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))




class Intervals:
	def __init__(self, filtration):
		complexes = filtration.complexes
		silent = filtration.silent
		self.epsilons = filtration.epsilons
		caller_dir = os.getcwd()
		os.chdir(SCRIPT_DIR)
		self.write_perseus_in_file(complexes, silent)
		self.call_perseus(silent)
		intervals = self.read_perseus_out_file(silent)
		self.birth_time, self.death_time = intervals[:, 0], intervals[:, 1]
		os.chdir(caller_dir)

	@staticmethod
	def write_perseus_in_file(filt_array, silent):
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

	@staticmethod
	def call_perseus(silent):
		os.chdir('perseus')
		for f in os.listdir('.'):
			if f.startswith('perseus_out'):
				os.remove(f)
		perseus_cmd = './{} nmfsimtop perseus_in.txt perseus_out'
		if sys.platform == 'linux' or sys.platform == 'linux2':
			perseus_cmd = perseus_cmd.format('perseusLin')
		elif sys.platform == 'darwin':  # macOS
			perseus_cmd = perseus_cmd.format('perseusMac')

		if silent:
			p = subprocess.Popen(perseus_cmd,
			                     shell=True, stdout=subprocess.PIPE)
			out, err = p.communicate()
		else:
			p = subprocess.Popen(perseus_cmd, shell=True)
			p.communicate()		# wait
		os.chdir('..')

	@staticmethod
	def read_perseus_out_file(silent):
		try:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				intervals = np.loadtxt('perseus/perseus_out_1.txt', ndmin=2)
		except IOError:
			intervals = np.empty((2, 0))
			if not silent: print "WARNING: no homology for this window"

		intervals[intervals == -1] = np.nan

		return intervals


class PDiagram:
	def __init__(self, intervals):
		self.epsilons = intervals.epsilons
		self.lim = self.epsilons[-1]
		self.mortal, self.immortal = self._build(intervals)


	def _t_to_eps(self, t):
		return self.epsilons[int(t - 1)]

	@staticmethod
	def _get_multiplicity(birth_e, death_e=None):
		if death_e is None:
			death_e = np.full_like(birth_e, -1)
		count = np.zeros_like(birth_e)
		for i, pt in enumerate(zip(birth_e, death_e)):
			for scanner_pt in zip(birth_e, death_e):
				if pt == scanner_pt:
					count[i] += 1
		return count

	def _build(self, intervals):
		birth_e_mor = []
		death_e_mor = []
		birth_e_imm = []

		for birth, death in zip(intervals.birth_time, intervals.death_time):
			if np.isnan(death):						        # immortal
				birth_e_imm.append(self._t_to_eps(birth))
			else:										    # mortal
				birth_e_mor.append(self._t_to_eps(birth))
				death_e_mor.append(self._t_to_eps(death))

		count_mor = self._get_multiplicity(birth_e_mor, death_e_mor)
		mortal = np.asarray([birth_e_mor, death_e_mor, count_mor]).T

		count_imm = self._get_multiplicity(birth_e_imm)
		immortal = np.asarray([birth_e_imm, count_imm]).T

		if len(mortal):
			# toss duplicates #
			mortal = np.vstack({tuple(row) for row in mortal})

		if len(immortal):
			# toss duplicates #
			immortal = np.vstack({tuple(row) for row in immortal})

		return mortal, immortal


class PRankFunction:
	def __init__(self, pd):
		self.epsilons = pd.epsilons
		self.data = self._build(pd)

	def _build(self, pd):

		num_div = len(self.epsilons)
		max_lim = pd.lim
		min_lim = 0

		x = y = np.linspace(min_lim, max_lim, num_div)
		xx, yy = np.meshgrid(x, y)

		grid_pts = zip(np.nditer(xx), np.nditer(yy))
		zz = np.zeros(len(grid_pts))
		for i, grid_pt in enumerate(grid_pts):
			if grid_pt[0] <= grid_pt[1]:
				for pt in pd.mortal:
					if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
						zz[i] += pt[2]
				for pt in pd.immortal:
					if pt[0] <= grid_pt[0]:
						zz[i] += pt[1]
			else:
				zz[i] = np.nan
		zz = np.reshape(zz, xx.shape)
		return zz



class Filtration:

	def __init__(self, traj, params, silent=False, save=True):
		"""
		Parameters
		----------
		traj : Trajectory
		params : dict
			see :py:func:`build_filtration.build_filtration`
		silent : bool, optional
			Suppress stdout
		save : bool or str, optional
			Save the filtration to file for later use. \n
			if ``save`` is a string:
				save filtration to ``save``\n
				should end with ``'.p'``
			elif ``save`` is True:
				save filtration to ``'PH/filtrations/filt.p'``\n
				filtration may be loaded by calling :py:func:`load_filtration`
				without specifying filename
			default: True

		"""

		self.name = traj.name
		self.fname = traj.fname
		self.ambient_dim = traj.dim
		self.params = params.copy()
		self.silent = silent

		caller_dir = os.getcwd()
		os.chdir(SCRIPT_DIR)

		arr = self._build(traj, params)

		self.epsilons = arr[3]
		self.witness_coords = arr[0]
		self.landmark_coords = arr[1]
		if not silent: print 'unpacking...'
		self.complexes = self._unpack_complexes(arr[2], silent)
		self.ambient_dim = self.witness_coords.shape[1]
		self.num_div = len(self.complexes)
		assert(self.num_div == len(self.epsilons))

		self._intervals = None
		self._PD = None
		self._PRF = None

		os.chdir(caller_dir)

		if save:
			if not silent: print 'pickling...'
			if isinstance(save, basestring):
				fname = save
			else:
				fname = 'PH/filtrations/filt.p'
			cPickle.dump(self, open(fname, 'wb'))

	def _build(self, traj, params):

		def compile_find_landmarks_c():
			if sys.platform == "linux" or sys.platform == "linux2":
				compile_str = find_landmarks_c_compile_str['linux']
			elif sys.platform == 'darwin':
				compile_str = find_landmarks_c_compile_str['macOS']
			else:
				print 'Sorry, PHETS requires linux or macOS.'
				sys.exit()
			subprocess.call(compile_str, shell=True)
			print '''find_landmarks compilation attempt complete. If 
			successful, please repeat your test. If problem persists, you will 
			need to tweak find_landmarks_c_compile_str in config.py to compile 
			find_landmarks.c on your system. '''
			sys.exit()

		silent = self.silent
		if not silent: print "building filtration..."

		if params['worm_length'] is None:
			params['worm_length'] = traj.data.shape[0]
		np.savetxt('temp/worm_data.txt', traj.data)
		start_time = time.time()

		try:
			if silent: blockPrint()
			filtration = build_filtration.build_filtration(
				'temp/worm_data.txt', params, silent=silent
			)
			if silent: enablePrint()
		except OSError:
			print "WARNING: invalid PH/find_landmarks binary. Recompiling..."
			compile_find_landmarks_c()

		os.remove('temp/worm_data.txt')

		witness_coords = np.array(filtration[1][1])
		landmark_coords = np.array(filtration[1][0])
		abstract_filtration = sorted(list(filtration[0]))
		epsilons = filtration[2]		# add to build_filtration return

		if not silent:
			t = time.time() - start_time
			print("build_filtration() time elapsed: {} seconds \n".format(t))
		return [witness_coords, landmark_coords, abstract_filtration, epsilons]


	def _unpack_complexes(self, simplex_list, silent):

		def group_by_birth_time(simplex_list):
			"""
			Reformats 1D list of SimplexBirth objects into 2D array of
			landmark_set lists, where rows are birth times
			"""
			ID_array = [[] for i in self.epsilons]
			for simplex in simplex_list:
				ID_array[simplex.birth_time].append(simplex.landmark_set)
			return ID_array


		def expand_to_2simplexes(ID_arr):
			"""
			for each k-simplex in filtration array, if k > 2, replaces
			with the component 2-simplexes(i.e. all length-3 subsets of
			landmark_ID_set)
			"""
			for row in ID_arr:
				expanded_row = []
				for landmark_ID_set in row:
					if len(landmark_ID_set) > 3:
						combinations = itertools.combinations
						expanded_set = combinations(landmark_ID_set, 3)
					else:
						expanded_set = [landmark_ID_set]
					expanded_row.extend(expanded_set)		# flatten
					# expanded_row.append(expanded_set)		# group by parent
				row[:] = expanded_row
			return np.asarray(ID_arr)

		def remove_duplicates_all(ID_arr):
			"""
			Omit simplexes that have been already added to the filtration or
			are repeated within a row
			"""
			all_tris = set()
			dups_count = 0
			for row in ID_arr:
				unique_row = []
				for tri in row:
					tr = frozenset(tri)
					if tr in all_tris:
						dups_count += 1
					else:
						unique_row.append(tri)
						all_tris.add(tr)

				row[:] = unique_row
				# print dups_count
			return np.asarray(ID_arr)


		def remove_duplicates_row(ID_arr):
			"""
			Omit duplicate simplexes within a row. It appears that there are
			in fact no duplicates of this nature
			"""
			dups_count = 0
			for row in ID_arr:
				sets_row = [frozenset(tri) for tri in row]
				unique_row = np.asarray(sets_row)
				dups_count += len(row) - len(unique_row)
				print dups_count
				row[:] = unique_row
			return ID_arr


		def count_triangles(ID_arr):
			tri_count = 0
			f = open('../output/run_info/num_triangles.txt', 'w')
			for i, row in enumerate(ID_arr):
				tris = [simp for simp in row if len(simp) == 3]
				tri_count += len(tris)
				f.write('frame {}: {}\n'.format(i + 1, tri_count))
			f.close()

		if not silent: print 'grouping by birth time...'
		ID_array = group_by_birth_time(simplex_list)    # 1d list -> 2d array
		if not silent: print 'expanding to 2-simplexes...'
		ID_array = expand_to_2simplexes(ID_array)
		if not silent: print 'removing duplicates...'
		ID_array = remove_duplicates_all(ID_array)
		if not silent: print 'counting triangles...'
		count_triangles(ID_array)
		return ID_array

	def intervals(self):
		if self._intervals is None:
			self._intervals = Intervals(self)
		return self._intervals

	def PD(self):
		"""
		if called for the first time:
			calls perseus, generates persistence diagram, sets, :py:attr:`_PD`,
			returns :py:attr:`_PD`
		else:
			returns :py:attr:`_PD`
		Returns
		-------
		PDData

		"""
		if self._PD is None:
			self._PD = PDiagram(self.intervals())
		return self._PD

	def PRF(self):
		"""
		if called for the first time:
			if :py:meth:`PD` has not been called:
				calls perseus, generates persistence diagram, sets,
				:py:attr:`_PD`, generates persistence rank function, sets,
				:py:attr:`_PRF`, returns :py:attr:`_PRF`
			else:
				generates persistence rank function, sets, :py:attr:`_PRF`,
				returns :py:attr:`_PRF`
		else:
			returns :py:attr:`_PRF`

		Parameters
		----------

		Returns
		-------
		array
			TODO: PRF class

		"""
		if self._PRF is None:
			self._PRF = PRankFunction(self.PD())
		return self._PRF


	def movie(self, filename, **kwargs):
		"""

		Parameters
		----------
		filename : str
			Output path/filename. Should end in '.mp4' or other movie format.
		color_scheme : str, optional
			``None``, ``'highlight new'``, or
			``('birth time gradient', cycles)`` where ``cycles`` is an ``int``
			default: ``None``
		camera_angle : array
			For 3D mode. (azimuthal, elevation) in degrees.\n
			default: (70, 45)
		alpha : float
			Opacity of simplexes\n
			default: 1
		dpi: int
			plot resolution -- dots per inch

		Returns
		-------
		None

		"""
		filtration_movie.build_movie(self, filename, **kwargs)

	def plot_complex(self, i, filename):
		"""
		plot complex at ith step of the filtration

		Parameters
		----------
		i : int
		filename : str
			Output path/filename. Should end in '.png' or other supported image
			format.\n

		Returns
		-------
		None

		"""
		raise NotImplemented

	def plot_PD(self, filename):
		"""
		plot the persistence diagram

		Parameters
		----------
		filename : str
			Output path/filename. Should end in '.png' or other supported image
			format.\n

		Returns
		-------
		None

		"""
		plots.PD_fig(self, filename)

	def plot_PRF(self, filename):
		"""
		plot the persistence rank function

		Parameters
		----------

		filename : str
			Output path/filename. Should end in '.png' or other supported image
			format.\n

		Returns
		-------
		None

		"""
		plots.PRF_fig(self, filename)


def load_filtration(filename=None):
	"""
	load a filtration from file

	Parameters
	----------
	filename : str
		Path/filename. Should end with ``'.p'``

	Returns
	-------

	"""
	print 'loading saved filtration...'
	if filename is None:
		filename= 'PH/filtrations/filt.p'
	filtration = cPickle.load(open(filename))
	return filtration
