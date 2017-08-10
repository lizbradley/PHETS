'''
original author: Jamie
'''

'''
Samantha Molnar
Began edits 10/24/16
'''



from sets import Set, ImmutableSet
import networkx as nx
import sys
import itertools
from heapq import heappush, heappop
import numpy as np
import math
import subprocess
import multiprocessing
from memory_profiler import profile
import os

# from Utilities import mem_profile
from config import MEMORY_PROFILE_ON

d = [] #this is where distance to all landmarks for each witness goes.  It is a list of'
def sort(i):
	d[i].sort()





# f = open("output/PH/build_filtration_memory.txt","wb")
#
# @profile(stream=f)
def build_filtration(input_file_name, parameter_set, silent=False):
	num_threads = 2
	global d
	d = []


	def get_param(key):
		return parameter_set.get(key)

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
	# always_euclidean = speed_amplify == orientation_amplify == stretch == ray_distance_amplify == 1.0 and use_hamiltonian == d_cov==0.
	# print "always_euclidean set to ", always_euclidean
	always_euclidean = get_param('always_euclidean')
	# use_hamiltonian = -5

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
	sort_output = get_param("sort_output")
	absolute = get_param("absolute")
	num_divisions = get_param("num_divisions")
	simplex_cutoff = get_param("simplex_cutoff")


	'''=============== This code written by Sam ======================'''

	## Read data into witness and landmark lists.
	witnesses = []
	landmarks = []
	landmark_indices = []
	ls = get_param("landmark_selector")
	downsample_rate = get_param("ds_rate")
	maxmin = False
	counter = 0
	for i in xrange(start):           #  Where to start reading data
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
	number_of_vertices = int(number_of_datapoints/downsample_rate)
	num_coordinates = len(witnesses[0])
	stop = start + counter

	if max_filtration_param < 0:
		if float(number_of_vertices) < abs(max_filtration_param) + 1:
			print '''ERROR: 'max_filtration_param' ({}) and number of landmarks ({})
			are incompatible. Try decreasing 'ds_rate' or increasing 'worm_length'.'''.format(max_filtration_param,
			number_of_vertices)
			sys.exit()

	num_threads = 2
	# for more information about these parameters type ./find_landmarks --help in the terminal
	# the distance calculations are calculated and outputted to a file called find_landmarks.txt
	print os.getcwd()

	if ls=="EST":
		if always_euclidean:
			if graph_induced:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-e {}".format(downsample_rate),
					"-x {}".format(d_cov),
					"-c",
					"-f {}".format(max_filtration_param)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
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
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-s {}".format(stretch),
					"-x {}".format(d_cov),
					"-e {}".format(downsample_rate),
					"-f {}".format(max_filtration_param)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
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
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-f {}".format(max_filtration_param)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
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
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-c",
					"-f {}".format(max_filtration_param)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
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
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch),
					"-f {}".format(max_filtration_param)
				]
			else:
				find_landmarks_cmd = [
					"./find_landmarks",
					"-n {}".format(num_threads),
					"-l {}".format(number_of_vertices),
					"-w {}-{}".format(start,stop),
					"-i{}".format(input_file_name),
					"-olandmark_outputs.txt",
					"-m {}".format(int(m2_d)),
					"-a {}".format(speed_amplify),
					"-y {}".format(orientation_amplify),
					"-h {}".format(use_hamiltonian),
					"-r {}".format(ray_distance_amplify),
					"-v {}".format(straight_VB),
					"-x {}".format(d_cov),
					"-s {}".format(stretch)
				]

	print find_landmarks_cmd

	if silent:
		p = subprocess.Popen(find_landmarks_cmd, stdout=subprocess.PIPE)
		out, err = p.communicate()
	else:
		p = subprocess.Popen(find_landmarks_cmd)
		p.communicate()

	## Build and sort distance matrix.
	landmarks_file = open("landmark_outputs.txt","rb")

	l = landmarks_file.readlines()
	sys.stdout.write("Reading in distance calculations...")
	sys.stdout.flush()
	landmark_index = 0
	for line in l:
		f = line.strip('\n')
		if "#" not in f:
			landmark = int(f.split(":")[0])

			distances = [float(i) for i in f.split(":")[1].split(",")]
			for witness_index in range(0,len(distances)):

				d[witness_index].append(LandmarkDistance(landmark_index,distances[witness_index]))
			landmarks.append(witnesses[landmark])
			landmark_indices.append(landmark)
			landmark_index+=1

	assert(len(d)>0)
	sys.stdout.write("done\n")
	sys.stdout.flush()



	sys.stdout.write("Sorting distances...")
	sys.stdout.flush()

	# p=multiprocessing.Pool(processes=4)   # commented out by Elliott 4/25

	inputs=[]
	for w in range(0,len(witnesses)):
		inputs.append(w)
		d[w].sort()

	# p.map(sort,inputs)  # was commented out as of 4/25
	# p.terminate()       # added by Elliott 4/25

	sys.stdout.write("done\n")
	sys.stdout.flush()
	assert len(landmarks) == number_of_vertices

	'''=============== End code written by Sam ======================'''

	'''============= Start code written by Elliott =================='''
	if graph_induced:
		# import matplotlib.pyplot as plt
		import pandas as pd

		g = nx.read_edgelist('edgelist.txt')

		closest_wits = np.loadtxt('closest_wits.txt', dtype='int')	# witness, landmark
		wit_coords = np.array(witnesses)
		land_coords = np.array(landmarks)

		# land = np.unique(closest_wits[:,1])

		# closest_wits = pd.DataFrame(closest_wits, columns=('witness', 'landmark'))
		# print closest_wits
		# closest_wits = closest_wits.sort_values(by='landmark')
		# print closest_wits
		#
		# closest_wits = closest_wits.values
		# print closest_wits



		# fig = plt.figure(figsize=(8, 8))
		# ax = fig.add_subplot(111)
		# ax.scatter(wit_coords[:, 0], wit_coords[:, 1], s=.1)
		# ax.scatter(land_coords[:, 0], land_coords[:, 1])
		# fig.savefig('veronoi_test.png')

	'''=============== End code written by Elliott =================='''




	print("Building filtration...")
	## Build filtration
	weak = get_param("weak")
	dimension_cutoff = get_param("dimension_cutoff")
	reentry_filter = get_param("reentry_filter")
	if get_param("connect_time_1_skeleton") or reentry_filter: # Connect time-1-skeleton
		for i in xrange(number_of_vertices - 1):
			filtration.add(SimplexBirth(ImmutableSet([i, i + 1]), 0, sort_output))
	use_cliques = get_param("use_cliques")
	use_twr = get_param("use_twr")

	print '%s' % use_twr
	if use_cliques: # AKA "Lazy" witness relation.
		g = nx.Graph()
		for l in xrange(number_of_vertices):
			g.add_node(l)
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
			filtration.add(SimplexBirth(clique, q, sort_output))
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
					filtration.add(SimplexBirth([d[witness_index][landmark_index].id_num for landmark_index in xrange(k + 1)], k, sort_output))
				else:
					if progress > 0:
						for base in itertools.combinations([d[witness_index][landmark_index].id_num for landmark_index in xrange(k)], min(k, dimension_cutoff)):
							new_subset = ImmutableSet(base + (d[witness_index][k].id_num,))
							filtration.add(SimplexBirth(new_subset, k, sort_output))
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
							filtration.add(SimplexBirth(new_subset, q, sort_output))
					add_simplex = True
				if (not use_cliques) and store_top_simplices and add_simplex and len(good_landmarks[witness_index])>= 2:
					filtration.add(SimplexBirth([good_landmarks[witness_index][i] for i in xrange(len(good_landmarks[witness_index]))], q, sort_output))
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
									filtration.add(SimplexBirth(new_subset, q, sort_output))
							add_simplex = True
						progress_index[witness_index] += 1
					else:
						break
				if (not use_cliques) and store_top_simplices and add_simplex and progress >= 2:
					list_o_landmarks = []
					for landmark_index in xrange(progress):
						list_o_landmarks.append(d[witness_index][landmark_index].id_num)
					#print 'At threshold %f, witness %d has landmark set %s' % (threshold, witness_index, str(list_o_landmarks))
					filtration.add(SimplexBirth([d[witness_index][landmark_index].id_num for landmark_index in xrange(progress)], q, sort_output))
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
		print("Done. Filtration contains %i top simplex birth events, with the largest epsilon equal to %f.\n" % (len(filtration), max_epsilon))
	else:
		max_sb_length = 0
		for sb in filtration:
			if len(sb.landmark_set) > max_sb_length:
				max_sb_length = len(sb.landmark_set)
		print("Done. Filtration contains %i top simplex birth events, with the largest one comprised of %i landmarks.\nMax filtration parameter: %s.\n" % (len(filtration), max_sb_length, max_filtration_param))

	## Write to output file
	output_file_name = get_param("out")

	if not output_file_name is None:
		output_file = open(output_file_name, "w")
		output_file.truncate()
		program = get_param("program")
		if dimension_cutoff is None:
			print("Writing filtration for input into %s..." % program)
			dimension_cutoff = number_of_vertices
		else:
			print("Writing filtration to file %s for input into %s, ignoring simplices above dimension %i..." % (output_file_name,program, dimension_cutoff))
		num_lines = 0
		if program == "Perseus":
			sets_printed_so_far = Set()
			num_lines = len(filtration) + 1
			output_file.write("1\n")
			list_filtration = None
			if (sort_output):
				list_filtration = list(filtration)
				list_filtration.sort()
			for simplex_birth in (list_filtration if sort_output else filtration):
				dimension = len(simplex_birth.landmark_set) - 1
				if dimension > dimension_cutoff:
					for subtuple in itertools.combinations(simplex_birth.landmark_set, dimension_cutoff + 1):
						subset = ImmutableSet(subtuple)
						if not ((subset, simplex_birth.birth_time) in sets_printed_so_far):
							output_file.write(str(dimension_cutoff) + " ")
							for landmark in subset:
								output_file.write(str(landmark + 1) + " ")
							output_file.write(str(simplex_birth.birth_time + 1) + "\n")
							sets_printed_so_far.add((subset, simplex_birth.birth_time))
				else:
					if not ((simplex_birth.landmark_set, simplex_birth.birth_time) in sets_printed_so_far):
						output_file.write(str(dimension) + " ")
						for landmark in (simplex_birth.sll if sort_output else simplex_birth.landmark_set):
							output_file.write(str(landmark + 1) + " ")
						output_file.write(str(simplex_birth.birth_time + 1) + "\n")
						sets_printed_so_far.add((simplex_birth.landmark_set, simplex_birth.birth_time))
		elif program == "PHAT":
			line_map = {}
			for i in xrange(number_of_vertices - 1):
				output_file.write("0\n")
				line_map[ImmutableSet([i])] = i
			output_file.write("0")
			line_map[ImmutableSet([number_of_vertices - 1])] = number_of_vertices - 1
			simultaneous_additions = []
			class Context: # Note: if upgrading to Python 3, one could just use the nonlocal keyword (see below comment).
				line_number = number_of_vertices
			list_filtration = list(filtration)
			list_filtration.sort()
			last_birth_time = 0
			def process_and_get_line_number(s):
				#nonlocal line_number
				if s in line_map:
					return line_map[s]
				else:
					dimension = len(s) - 1
					if dimension > dimension_cutoff:
						for subset in itertools.combinations(s, dimension_cutoff + 1): # Take all subsets of size dimension_cutoff + 1
							process_and_get_line_number(ImmutableSet(subset))
					elif dimension > 0:
						subsets_line_numbers = []
						for e in s:
							subsets_line_numbers.append(process_and_get_line_number(ImmutableSet(s - Set([e]))))
						output_file.write("\n" + str(dimension))
						for l in subsets_line_numbers:
							output_file.write(" " + str(l))
						line_map[s] = Context.line_number
						Context.line_number += 1
						return Context.line_number - 1
					else:
						raise Exception("Should have already added single point for base case: " + str(s))
			for simplex_birth in list_filtration:
				if simplex_birth.birth_time > last_birth_time:
					simultaneous_additions.append((Context.line_number - 1, last_birth_time + 1)) # Every line up to and including that line number (indexing starts at 0) had that birth time or earlier (indexing starts at 1)
					last_birth_time = simplex_birth.birth_time
				process_and_get_line_number(simplex_birth.landmark_set)
			simultaneous_additions.append((sys.maxsize, last_birth_time))
			output_file.write("\n\n# Simultaneous additions: Every line up to and including __ (indexing starts at 0) has birth time __ (or earlier).")
			for addition in simultaneous_additions:
				output_file.write("\n# %20i %20i" % addition)
			extra_data = (extra_data[0], extra_data[1], simultaneous_additions)
			num_lines = Context.line_number
		else:
			raise Exception("Only supported programs are 'Perseus' and 'PHAT'")
		output_file.close()
		print("Done. File contains %i lines.\n" % num_lines)
	print("Filtration has been successfully built!\n")
	return (filtration, extra_data + (max_filtration_param,), epsilons)

class SimplexBirth:
	# An object that contains a set of landmarks that define a simplex, and a birth time measured in integer units.
	# IMPORTANT: two SimplexBirths are considered equal iff they have the same landmark set, regardless of birth time.

	include_birth_time = False

	def __init__(self, landmark_list, birth_time, keep_sorted_list):
		if (keep_sorted_list):
			self.sll = sorted(landmark_list)
		else:
			self.sll = None
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
				if self.sll is None:
					return 0
				for i in xrange(len(self.sll)):
					if self.sll[i] < other.sll[i]:
						return -1
					elif self.sll[i] > other.sll[i]:
						return 1
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
