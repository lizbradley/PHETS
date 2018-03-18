"""
original authors: Jamie and Nikki
edits by Samantha Molnar and Elliott Shugerman
"""

from sets import Set, ImmutableSet
import networkx as nx
import sys
import itertools
import numpy as np
import math
import subprocess
from utilities import make_dir, get_label


# f = open("output/run_info/build_filtration_memory.txt","wb")
# @profile(stream=f)
def build_filtration(input_file_name, parameter_set, silent=False):
	"""
	Parameters
	----------
	input_file_name : str
	parameter_set : dict
		Options for filtration and landmark selection. Defaults are set in
		``config.py``\n
		GENERAL:
			num_divisions
				Number of (epsilon) steps in filtration. The filtration
				parameter will be divided up equally in the interval
				[min_filtration_param, max_filtration_param].\n
				default: 50
			max_filtration_param
				The maximum value for the filtration parameter. If it is a
				negative integer, -x, the program will automatically choose the
				max filtration parameter such that the highest dimensional
				simplex constructed is of dimension x - 1.\n
				default: -20
			min_filtration_param
				The minimum value for the filtration parameter. Zero is usually
				fine.\n
				default: 0
			start
				How many lines to skip in the input file before reading data
				in.\n
				default: 0
			worm_length
				How many witnesses the program will read from the data file.
				If set to None, the program will read the file to the end. In
				general, a reasonable cap we have found is 10,000 witnesses
				and 200 or less landmarks.\n
				default: None
			ds_rate
				The ratio of number of witnesses / number of landmarks.\n
				default: 50
			landmark_selector
				 "maxmin" How the landmarks are selected from among the
				 witnesses. Only options are "EST" for equally spaced in time
				 and "maxmin" for a max-min distance algorithm.\n
				 default: "maxmin"
		WITNESS RELATION:
			absolute
				The standard fuzzy witness relation says that a witness
				witnesses a simplex if the distance from the witness to each of
				the landmarks is within epsilon *more* than the distance to the
				closest landmark. If using the absolute relation, the closest
				landmark is dropped from the calculation, and the distance from
				a witness to each of the landmarks must be within epsilon of
				zero.\n
				default: False
			use_cliques
				If this is set to True, than witnesses are only used to
				connect edges, and higher simplices (faces, solids, etc.) are
				inferred from the 1-skeleton graph using the Bron-Kerbosch
				maximal clique finding algorithm. This can be useful in
				reducing noise if several of the false holes are triangles.\n
				default: False
			simplex_cutoff
				If not equal to zero, this caps the number of landmarks a
				witness can witness. Note: this does not effect automatic
				max_filtration_param selection.\n
				default: 0
			weak
				Uses a completely different relation. The filtration
				parameter k specifies that each witness will witness a
				simplex of its k-nearest neighbors. If this relation is used,
				max_filtration_param should be a positive integer,
				and num_divisions and min_filtration_param will be ignored.\n
				default: False
			use_twr
				Uses a completely different algorithm. TODO: insert your
				description here. Note: this works best with EST landmark
				selection. If max-min is used, be sure to set
				time_order_landmarks to True.\n
				default: False
		DISTANCE DISTORTIONS:
			d_speed_amplify
				The factor by which distances are divided if the witness is
				at a relatively high speed.\n
				default: 1
			d_orientation_amplify
				The factor by which distances are divided if the witness and
				the landmark are travelling in similar directions.\n
				default: 1
			d_stretch
				The factor by which distances are divided if the vector from
				the witness to the landmark is in a similar direction (
				possibly backwards) as the direction in which time is flowing
				at the witness.\n
				default: 1
			d_ray_distance_amplify
				TODO: change this parameter. Right now, as long as the number
				is not 1, this will multiply the distance between two points
				by the distance between the closest points on the
				parameterized rays.\n
				default: 1
			d_use_hamiltonian
				If this is not zero, this will override all the above
				distortions. Distance will be computed using not only
				position coordinates, but also velocity coordinates. Velocity
				componnents are scaled by the value of this parameter (before
				squaring). If the value is negative, than the absolute value
				of the parameter is used, but the unit velocities are used
				instead of the actual velocities.\n
				default: 0
			use_ne_for_maxmin
				Whether or not to apply the above distance distortions to the
				max-min landmark selection (not recommended). Has no effect
				if landmark selector is EST.\n
				default: False
		MISC:
			connect_time_1_skeleton
				If this is set to True, then on the first step of the
				filtration, each landmark will be adjoined by an edge to the
				next landmark in time. Note: this works best with EST
				landmark selection. If max-min is used, be sure to set
				time_order_landmarks to True.\n
				default: False
			reentry_filter
				Attempts to limit high dimensional simplices by requiring
				that landmarks get far away then come back. This only works
				if using cliques. Note: this works best with EST landmark
				selection. If max-min is used, be sure to set
				time_order_landmarks to True.\n
				default: False
			dimension_cutoff
				Simplexes with dimension greater than the dimension cuttoff
				will be seperated into their lower dimensional subsets when
				writing to the output file. This is very handy, as both
				Perseus and PHAT seem to take exponential time as a function
				of the dimension of a simplex. The caveat is that all
				homology greater than or equal to the dimension cutoff will
				be inacurate. Thus, if one cares about Betti 2, dimension
				cutoff should be at least 3.\n
				still valid / in use ??? setting to 0 doesn't affect tests\n
				default: 2
			store_top_simplices
				If there is a dimension cutoff in use, this parameter
				determines at which point in the process the simplices are
				decomposed. By setting this to False, smaller simplices will
				be stored when they are discovered. This makes the output
				file a bit smaller, but takes a bit longer. The results will
				be left unchanged.\n
				default: True
	silent : bool
		Suppress stdout

	Returns
	-------
	array
		(simplexes, (landmarks, witnesses), eps)

	"""
	num_threads = 2
	d = []

	def get_param(key):
		return parameter_set[key]

	input_file = open(input_file_name)
	speed_amplify = float(get_param("d_speed_amplify"))
	orientation_amplify = float(get_param("d_orientation_amplify"))
	stretch = float(get_param("d_stretch"))
	ray_distance_amplify = get_param("d_ray_distance_amplify")
	use_hamiltonian = get_param("d_use_hamiltonian")
	print "use hamiltonian set to ", use_hamiltonian
	m2_d = float(get_param("m2_d"))
	straight_VB = float(get_param("straight_VB"))
	d_cov = get_param("d_cov")
	graph_induced = get_param("graph_induced")

	always_euclidean = speed_amplify == orientation_amplify == stretch == ray_distance_amplify == use_hamiltonian == 1.0 and d_cov == 0.
	print 'always_euclidean: {}'.format(always_euclidean)
	filtration = Set()
	extra_data = None
	min_filtration_param = float(get_param("min_filtration_param"))
	max_filtration_param = float(get_param("max_filtration_param"))
	print "Max filtration parameter is ", max_filtration_param
	if max_filtration_param < 0 and min_filtration_param != 0:
		raise Exception("Argument 'min_filtration_param' is incompatible with automatic max_filtration_param selection.")
	number_of_vertices = 0
	start = get_param("start")
	worm_length = get_param("worm_length")
	store_top_simplices = get_param("store_top_simplices")
	absolute = get_param("absolute")
	num_divisions = get_param("num_divisions")
	simplex_cutoff = get_param("simplex_cutoff")
	file_suffix = get_label();
        intermediate_path = '../output/phomology/' + file_suffix
	make_dir(intermediate_path);
	landmark_out_str = intermediate_path + "/landmark_outputs.txt"

	##################### begin edits by Sam and Elliott ######################

	## Read data into witness and landmark lists.
	witnesses = []
	landmarks = []

	counter = 0
	for i in xrange(start):         # Where to start reading data
		input_file.readline()
		counter+=1
	landmark_indices=[]

	for line in input_file.read().split("\n"):
		if line != "" and counter>=start:
			string_witness = line.split(" ")
			witness = []
			d.append([])
			for coordinate in string_witness:
				if coordinate != "":
					witness.append(float(coordinate))
			witnesses.append(witness)
			counter += 1
			if counter == worm_length:
				break

	number_of_datapoints = len(witnesses)
	downsample_rate = get_param("ds_rate")
	# if downsample_rate < 0:     # num landmarks
	# 	if worm_length:
	# 		downsample_rate = worm_length / abs(downsample_rate)
	# 	else: downsample_rate = number_of_datapoints / abs(downsample_rate)
        
	number_of_vertices = int(number_of_datapoints/downsample_rate)

	print 'number of witnesses: {}'.format(number_of_datapoints)
	print 'number of landmarks: {}'.format(number_of_vertices)
	print 'downsample rate: {}'.format(downsample_rate)

	stop = start + counter

	if max_filtration_param < 0:
		if float(number_of_vertices) < abs(max_filtration_param) + 1:
			msg = ''''max_filtration_param' ({}) and number of landmarks
			 ({}) are incompatible. Try decreasing 'ds_rate' or increasing 
			 'worm_length'.'''.format(max_filtration_param, number_of_vertices)
			raise Exception(msg)

	ls = get_param("landmark_selector")

	if ls=="EST":
		if always_euclidean:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m }".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-e {}".format(downsample_rate),
					"-x {}".format(d_cov),
					"-c",
					"-f {}".format(max_filtration_param),
					"-d {}".format(num_divisions)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-e {}".format(downsample_rate),
					"-x {}".format(d_cov),
					"-c"
				]
		else:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-x {}".format(d_cov),
					"-e {}".format(downsample_rate),
					"-f {}".format(max_filtration_param),
					"-d {}".format(num_divisions)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-x {}".format(d_cov),
					"-e {}".format(downsample_rate)
				]
	else:
		if always_euclidean and m2_d!=0:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-f {}".format(max_filtration_param),
					"-d {}".format(num_divisions)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch)
				]
		elif always_euclidean:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-c",
					"-f {}".format(max_filtration_param),
					"-d {}".format(num_divisions)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-c"
				]
		else:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-f {}".format(max_filtration_param),
					"-d {}".format(num_divisions)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i {}".format(input_file_name),
					"-o " + landmark_out_str,
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch)
				]

	arg_idxs = {        # see find_landmarks.c:110
		'i': 1,
		'o': 2,
		'l': 3,
		'w': 4,
		'e': 5,
		't': 6,
		'q': 7,
		'a': 8,
		'y': 9,
		'h': 10,
		'm': 11,
		'r': 12,
		'n': 13,
		'v': 14,
		's': 15,
		'c': 16,
		'x': 17,
		'f': 18,
		'd': 19,
	}

	arg_idxs = {key: value - 1 for key, value in arg_idxs.iteritems()}

	cmd = list(find_landmarks_cmd)
	cmd.remove('./find_landmarks')

	switches = np.zeros(len(arg_idxs), dtype=int)
	values = np.empty(len(arg_idxs), dtype=object)

	for c in cmd:
		arg, param = c[1], c[2:].strip()
		arg_idx = arg_idxs[arg]
		switches[arg_idx] = 1 
		values[arg_idx] = param

	values = [None if v == '' else v for v in values]
	switch_file = intermediate_path + '/find_landmark_arg_switches.txt'
	val_file = intermediate_path + '/find_landmark_arg_vals.txt'
	np.savetxt(switch_file, switches, fmt='%i')
	np.savetxt(val_file, values, fmt='%s')

	if silent:
		p = subprocess.Popen(["./find_landmarks", intermediate_path], stdout=subprocess.PIPE)
		out, err = p.communicate()
	else:
		p = subprocess.Popen(['./find_landmarks', intermediate_path])
		p.communicate()

	if m2_d!=0:
		number_of_datapoints = int(number_of_datapoints-m2_d)

	## Build and sort distance matrix.
	landmarks_file = open(landmark_out_str,"rb")

	lines = landmarks_file.readlines()
	sys.stdout.write("Reading in distance calculations...")
	sys.stdout.flush()
	landmark_index = 0
	for line in lines:
		f = line.strip('\n')
		if "#" not in f:
			landmark = int(f.split(":")[0])

			distances = [float(i) for i in f.split(":")[1].split(",")]

			for witness_index in range(0, len(distances)):
				d[witness_index].append(
					LandmarkDistance(landmark_index, distances[witness_index])
				)

			landmarks.append(witnesses[landmark])
			landmark_indices.append(landmark)
			landmark_index += 1

	assert(len(d)>0)
	sys.stdout.write("done\n")
	sys.stdout.flush()



	sys.stdout.write("Sorting distances...")
	sys.stdout.flush()


	inputs = []
	for w in range(0, len(witnesses)):
		inputs.append(w)
		d[w].sort()


	sys.stdout.write("done\n")
	sys.stdout.flush()
	assert len(landmarks) == number_of_vertices


	if graph_induced:
		w2l_id_dict = {}
		for land_id, wit_id in enumerate(landmark_indices):
			w2l_id_dict[wit_id] = land_id

		def wit_ids_2_land_ids(simplex):
			return [w2l_id_dict[wit_id] for wit_id in simplex]

		with open(intermediate_path + '/gi_edge_filtration.txt', 'r') as f:
			lines = f.readlines()

		eps, filt_diffs = [], []
		for line in lines:
			e, filt_diff_str = line.split(': ')
			eps.append(float(e))

			fd_str_arr = filt_diff_str.split(' ')
			fd_str_arr = [
				simp_str.replace('[', '').replace(']', '')
				for simp_str in fd_str_arr
			]
			filt_diff = [
				np.fromstring(simp_str, sep=',', dtype=int)
				for simp_str in fd_str_arr
			]
			filt_diffs.append(filt_diff)

		filt_diffs = np.asarray(filt_diffs)

		for i, row in enumerate(filt_diffs):
			for j, simp in enumerate(row):
				filt_diffs[i][j] = wit_ids_2_land_ids(simp)

		simplexes = []
		for i, row in enumerate(filt_diffs):
			complex = [SimplexBirth(ids, i) for ids in row]
			simplexes.extend(complex)

		return simplexes, (landmarks, witnesses), eps


	###################### end edits by Sam and Elliott #######################


	print("Building filtration...")
	## Build filtration
	weak = get_param("weak")
	dimension_cutoff = get_param("dimension_cutoff")
	reentry_filter = get_param("reentry_filter")
	if get_param("connect_time_1_skeleton") or reentry_filter: # Connect time-1-skeleton
		for i in xrange(number_of_vertices - 1):
			filtration.add(SimplexBirth(ImmutableSet([i, i + 1]), 0))
	use_cliques = get_param("use_cliques")
	use_twr = get_param("use_twr")

	print '%s' % use_twr
	if use_cliques: # AKA "Lazy" witness relation.
		g = nx.Graph()
		for line in xrange(number_of_vertices):
			g.add_node(line)

	def filter_and_build():
		g2 = None
		if reentry_filter:
			g2 = g.copy()
			to_remove = Set()
			for l1 in xrange(number_of_vertices):
				l2 = l1 + 2
				while l2 < number_of_vertices and g2.has_edge(l1, l2):
					to_remove.add(ImmutableSet([l1, l2]))
					l2 += 1
			for edge in to_remove:
				g2.remove_edge(*tuple(edge)) # May cause weird things to happen because removing edges doesn't remove them from the filtration.
		else:
			g2 = g
		for clique in nx.find_cliques(g2):
			filtration.add(SimplexBirth(clique, q))

	if weak: # Builds filtration based on k nearest neighbors.
		if max_filtration_param % 1 != 0:
			raise Exception("Argument 'max_filtration_param' must be an integer if using the weak witness relation.")
		max_filtration_param = int(max_filtration_param)
		for k in xrange(int(math.fabs(max_filtration_param))):
			for witness_index in xrange(number_of_datapoints):
				if use_cliques:
					for i in xrange(k):
						g.add_edge(d[witness_index][i].id_num, d[witness_index][k].id_num)
				elif store_top_simplices:
					filtration.add(SimplexBirth(
						[d[witness_index][landmark_index].id_num for
						 landmark_index in xrange(k + 1)], k))
				else:
					if progress > 0:
						for base in itertools.combinations([d[witness_index][landmark_index].id_num for landmark_index in xrange(k)], min(k, dimension_cutoff)):
							new_subset = ImmutableSet(base + (d[witness_index][k].id_num,))
							filtration.add(SimplexBirth(new_subset, k))
			if use_cliques:
				filter_and_build()
	if use_twr:
		print 'Using TWR'
		if max_filtration_param < 0: # Automatically determine max.
			depth = int(-max_filtration_param)
			min_distance = None
			for w in xrange(number_of_datapoints):
				new_distance = d[w][depth].distance - (0 if absolute else d[w][0].distance)
				if min_distance is None or new_distance < min_distance:
					min_distance = new_distance
			max_filtration_param = min_distance
		print 'The max_filtration_param is %d ' % max_filtration_param
		step = float(max_filtration_param - min_filtration_param)/float(num_divisions) # Change in epsilon at each step.
		print 'The step size is %f ' % step
		print 'There will be %d steps in the filtration' % num_divisions
		progress_index = [0]*number_of_datapoints
		done = False

		good_landmarks = [[] for x in range(number_of_datapoints)]

		epsilons = []
		for q in xrange(num_divisions):
			threshold = (max_filtration_param if q == num_divisions - 1 else float(q + 1)*step + min_filtration_param)
			print 'The threshold is currently %f' % threshold
			epsilons.append(threshold)
			Pre_landmarks = []
			for witness_index in xrange(number_of_datapoints):
				pre_landmarks = []
				add_simplex = False
				progress = 0
				while True:
					progress = progress_index[witness_index]
					if simplex_cutoff > 0 and progress >= simplex_cutoff:
						break
					if progress == number_of_vertices:
						done = True
						break

					if d[witness_index][progress].distance < threshold + (0 if absolute else d[witness_index][0].distance):
						pre_landmarks.append(d[witness_index][progress].id_num) # PRE_LANDMARKS CONTAINS ID NUMBER
						progress_index[witness_index] += 1
					else:

						pre_landmarks_size = len(pre_landmarks)
						pre_landmarks_string = str(pre_landmarks) # MAKE LIST TO STRING
						print 'At threshold value %f, witness %d has %d associated landmarks: %s ' % (threshold, witness_index, pre_landmarks_size, pre_landmarks_string)
						break
				Pre_landmarks.append(pre_landmarks)
				Pre_landmarks_size = len(Pre_landmarks)



			for witness_index in xrange(number_of_datapoints - downsample_rate):
				if len(Pre_landmarks[witness_index]) == 1:
					set_range = 1
				else:
					set_range = len(Pre_landmarks[witness_index])
				for k in range(set_range):
					current_pre_landmark = Pre_landmarks[witness_index][k]
					next_pre_landmark = Pre_landmarks[witness_index][k]+1 # CHECKS ONE STEP UP FROM ID NUMBER *** SEE JAMIE'S COMMENTS ON SUCH OR CLARIFY ***
					#	print 'current pre landmark = %d , next pre landmark = %d' % (current_pre_landmark, next_pre_landmark)
					check_pre_landmark = str(Pre_landmarks[witness_index + downsample_rate]) # HMMMMM
					#		print 'We are considering the fate of landmark %d witnessed by witness %d...' % (current_pre_landmark, witness_index)
					#		print 'Should witness %d not witness landmark %d, it will be GONE!' % (witness_index + downsample_rate, current_pre_landmark + 1,)
					#		print 'Witness %d has landmark set %s' % (witness_index + downsample_rate, check_pre_landmark)
					print (Pre_landmarks[witness_index][k]) in Pre_landmarks[witness_index]
					if (Pre_landmarks[witness_index][k]+ 1) in Pre_landmarks[witness_index + downsample_rate]: # change from 1, downsample_rate to 0 to test!
						good_landmarks[witness_index].append(Pre_landmarks[witness_index][k])
				print 'Up to threshold value %f, witness %d has landmark set %s' % (threshold, witness_index, str(good_landmarks[witness_index]))
				if use_cliques:
					for i in xrange(len(good_landmarks[witness_index])):
						for j in xrange(i+1,len(good_landmarks[witness_index])):
							g.add_edge(good_landmarks[witness_index][i], good_landmarks[witness_index][j])
				else:
					if not store_top_simplices and len(good_landmarks[witness_index]) > 0:
						for base in itertools.combinations(good_landmarks[witness_index], min(len(good_landmarks[witness_index]), dimension_cutoff)):
							new_subset = ImmutableSet(base + (good_landmarks[witness_index][i],))
							filtration.add(SimplexBirth(new_subset, q))
					add_simplex = True
				if (not use_cliques) and store_top_simplices and add_simplex and len(good_landmarks[witness_index])>= 2:
					filtration.add(SimplexBirth(
						[good_landmarks[witness_index][i] for i in
						 xrange(len(good_landmarks[witness_index]))], q))
				if done:
					break
			if use_cliques:
				filter_and_build()
			if done:
				break
			#	print 'We are done with threshold %f' % threshold
	else:
		if max_filtration_param < 0: # Automatically determine max.
			depth = int(-max_filtration_param)
			min_distance = None
			for w in xrange(number_of_datapoints):
				new_distance = d[w][depth].distance - (0 if absolute else d[w][0].distance)
				if min_distance is None or new_distance < min_distance:
					min_distance = new_distance
					if min_distance ==0:
						print "witness ",w
			max_filtration_param = min_distance

		step = float(max_filtration_param - min_filtration_param)/float(num_divisions) # Change in epsilon at each step.
		progress_index = [0]*number_of_datapoints
		done = False
		epsilons = []
		for q in xrange(num_divisions):
			threshold = (max_filtration_param if q == num_divisions - 1 else float(q + 1)*step + min_filtration_param)
			print 'The threshold is currently %f' % threshold
			epsilons.append(threshold)
			for witness_index in xrange(number_of_datapoints):
				add_simplex = False
				progress = 0
				while True:
					progress = progress_index[witness_index]
					if simplex_cutoff > 0 and progress >= simplex_cutoff:
						break
					if progress == number_of_vertices:
						done = True
						break
					if d[witness_index][progress].distance < threshold + (0 if absolute else d[witness_index][0].distance):
						if use_cliques:
							for i in xrange(progress):
								g.add_edge(d[witness_index][i].id_num, d[witness_index][progress].id_num)
						else:
							if not store_top_simplices and progress > 0:
								for base in itertools.combinations([d[witness_index][landmark_index].id_num for landmark_index in xrange(progress)], min(progress, dimension_cutoff)):
									new_subset = ImmutableSet(base + (d[witness_index][progress].id_num,))
									filtration.add(SimplexBirth(new_subset, q))
							add_simplex = True
						progress_index[witness_index] += 1
					else:
						break
				if (not use_cliques) and store_top_simplices and add_simplex and progress >= 2:
					list_o_landmarks = []
					for landmark_index in xrange(progress):
						list_o_landmarks.append(d[witness_index][landmark_index].id_num)
					#print 'At threshold %f, witness %d has landmark set %s' % (threshold, witness_index, str(list_o_landmarks))
					filtration.add(SimplexBirth(
						[d[witness_index][landmark_index].id_num for
						 landmark_index in xrange(progress)], q))
				if done:
					break
			if use_cliques:
				filter_and_build()
			if done:
				break


	extra_data = (landmarks, witnesses)
	if weak:
		max_epsilon = 0.0
		for w in xrange(number_of_datapoints):
			#print("type: %s" % type(d[w][max_filtration_param - 1].distance))
			#print("value: %f" % d[w][max_filtration_param - 1].distance)
			if (d[w][max_filtration_param - 1].distance) > max_epsilon:
				max_epsilon = d[w][max_filtration_param - 1].distance
		print(
			"Done. Filtration contains %i top simplex birth events, with the\n"
			"largest epsilon equal to %f.\n" % (len(filtration), max_epsilon)
		)
	else:
		max_sb_length = 0
		for sb in filtration:
			if len(sb.landmark_set) > max_sb_length:
				max_sb_length = len(sb.landmark_set)
		print(
			"Done. Filtration contains %i top simplex birth events, with the\n"
			"largest epsilon equal to %f.\n" % (len(filtration), max_sb_length)
		)

		print 'Max filtration parameter: {}'.format(max_filtration_param)

	print("Filtration has been successfully built!\n")
	return (filtration, extra_data + (max_filtration_param,), epsilons)

class SimplexBirth:
	# An object that contains a set of landmarks that define a simplex, and a birth time measured in integer units.
	# IMPORTANT: two SimplexBirths are considered equal iff they have the same landmark set, regardless of birth time.

	include_birth_time = False

	def __init__(self, landmark_list, birth_time):
		self.landmark_set = ImmutableSet(landmark_list)
		self.birth_time = birth_time

	def __eq__(self, other): # For hashing
		if SimplexBirth.include_birth_time:
			return self.birth_time == other.birth_time and self.landmark_set.__eq__(other.landmark_set)
		else:
			return self.landmark_set.__eq__(other.landmark_set)

	def __cmp__(self, other): # For sorting
		if self.birth_time < other.birth_time:
			return -1
		elif self.birth_time > other.birth_time:
			return 1
		else:
			if len(self.landmark_set) < len(other.landmark_set):
				return -1
			elif len(self.landmark_set) > len(other.landmark_set):
				return 1
			else:
				return 0

	def __hash__(self):
		return self.landmark_set.__hash__()

class LandmarkDistance: # An object that contains both the distance to the landmark and that landmark's ID number.

	def __init__(self, id_num, distance):
		self.id_num = id_num
		self.distance = float(distance)

	def __cmp__(self, other):
		if self.distance < other.distance:
			return -1
		elif self.distance > other.distance:
			return 1
		else:
			return 0

	def __le__(self, other): # Called in heap operations.
		return self.distance <= other.distance

	def __str__(self):
		return "(%i, %.9f)" % (self.id_num, self.distance)

	__repr__ = __str__
