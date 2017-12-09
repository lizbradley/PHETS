""" old code for 3D filt movies used to work (if mayavi was installed, tricky
on OSX with matplotlib 2), then was deleted in leu of gnuplot. salvaged from
commit history. would be cool to reimplement and add a switch in config.py for
3D filt movie backend: gnuplot or mayavi"""

def unpack_complex_data_3D(complex_list):

	def group_by_birth_time(complex_ID_list):
		"""Reformats 1D list of SimplexBirth objects into 2D array of
		landmark_set lists, where 2nd index is  birth time (? see below)"""

		# TODO: ensure that if a time t has no births, the row t is empty/skipped

		complex_ID_array = []	# list of complex_at_t lists
		complex_at_t = []	# list of simplices with same birth_time
		i = 0
		time = 0
		list_length = len(complex_ID_list)
		while i < list_length:
			birth_time = complex_ID_list[i].birth_time
			if birth_time == time:
				complex_at_t.append(complex_ID_list[i].landmark_set)
				i += 1
			else:
				complex_ID_array.append(complex_at_t)
				complex_at_t = []
				time += 1
		return complex_ID_array

	def expand_to_2simplexes(ID_array):
		"""accepts a k-simplex and, if k > 2, returns the component 2-simplexes
		(i.e. all length-3 subsets of landmark_ID_set), else returns input"""
		for row in ID_array:
			expanded_row = []
			for landmark_ID_set in row:
				expanded_set = itertools.combinations(landmark_ID_set, 3) \
					if len(landmark_ID_set) > 3 else [tuple(landmark_ID_set)]
				expanded_row.extend(expanded_set)
			row[:] = expanded_row

	def separate_by_k (array):
		lines = []
		triangles = []
		for row in array:
			lines_row = []
			triangles_row = []
			for simplex in row:
				if len(simplex) == 2:
					lines_row.append(simplex)
				else: #if len(simplex) == 3:
					triangles_row.append(simplex)
			triangles.append(triangles_row)
			lines.append(lines_row)
		return [lines, triangles]

	ID_array = group_by_birth_time(complex_list)
	expand_to_2simplexes(ID_array)
	complexes = separate_by_k(ID_array)
	return complexes

def load_data():
	witness_data = np.load('PersistentHomology/temp_data/witness_coords.npy')
	landmark_data = np.load('PersistentHomology/temp_data/landmark_coords.npy')
	complex_data = np.load('PersistentHomology/temp_data/complexes.npy')
	ambient_dim = len(witness_data[1])
	return ambient_dim, [witness_data, landmark_data, complex_data]


def make_frame3D(birth_time, camera_angle=(135, 55), hide_1simplexes=False, alpha=.7, color_scheme='none'):
	ambient_dim, filt_data = load_data()

	def get_simplex_color(scheme, birth_time, current_birth_time, max_birth_time):
		"""helper for plot_complex()"""
		if scheme == 'none':
			color = (.4, .6, .8)
		elif scheme == 'highlight new':
			color = (1, 0, 0) if birth_time == current_birth_time - 1 else (0, 0, 1)
		elif scheme == 'birth_time gradient':
			prog = birth_time / float(max_birth_time)
			color = (0, prog, 1 - prog)
		else:
			print 'error:', scheme, 'is not a valid color scheme'
		return color

	def plot_witnesses(witness_data):
		x = witness_data[:, 0]
		y = witness_data[:, 1]
		z = witness_data[:, 2]
		s = np.ones(len(x)) * .005
		# mlab.points3d(x, y, z, mode='point', color=(0, 0, 0))
		mlab.points3d(x, y, z, s, mode='sphere', color=(0, 0, 0), scale_factor=1)


	def plot_landmarks(landmark_data):
		x = landmark_data[:, 0]
		y = landmark_data[:, 1]
		z = landmark_data[:, 2]
		mlab.points3d(x, y, z, scale_factor=.02, color=(0, 0, 1))

	def plot_complex(complex_data, current_birth_time, landmark_data):  # how to specify color per simplex??
		"""plots plots all simplices with birth time =< birth_time"""
		max_birth_time = len(complex_data) - 1
		birth_time = 0
		while birth_time < current_birth_time:
			# color = get_simplex_color_3D(color_scheme, birth_time, current_birth_time, max_birth_time, landmark_data)
			# color = ((<float> for id in triangle_ID) for triangle_ID in triangle_IDs)
			# then, in triangular_mesh(...., scalers=color)

			color = get_simplex_color(color_scheme, birth_time, current_birth_time, max_birth_time)

			triangle_IDs =  complex_data[1][birth_time]
			x = landmark_data[:, 0]
			y = landmark_data[:, 1]
			z = landmark_data[:, 2]


			mlab.triangular_mesh(x, y, z, triangle_IDs, color=color, opacity=alpha, representation='surface')
			mlab.triangular_mesh(x, y, z, triangle_IDs, color=(0, 0, 0), representation='wireframe')

			if hide_1simplexes == False:
				lines = complex_data[0][birth_time]
				for simplex_IDs in lines:
					ID_coords = np.array([landmark_data[simplex_IDs[0]], landmark_data[simplex_IDs[1]]])
					x = ID_coords[:, 0]
					y = ID_coords[:, 1]
					z = ID_coords[:, 2]
					mlab.plot3d(x, y, z, tube_radius=None, color=(0,0,0))
					# mlab.pipeline.line_source(x, y, z, figure=fig)

			birth_time += 1

	mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
	mlab.view(azimuth=camera_angle[0], elevation=camera_angle[1], focalpoint='auto', distance='auto')


	filt_data[2] = unpack_complex_data_3D(filt_data[2])

	plot_witnesses(filt_data[0])
	plot_landmarks(filt_data[1])
	plot_complex(filt_data[2], birth_time, filt_data[1])

	# mlab.savefig(filename='frames/sub_img%03d.png' % i)   # for debugging
	mlab.show()

