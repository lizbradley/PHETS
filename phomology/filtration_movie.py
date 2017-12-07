import io, subprocess, sys, os

import matplotlib.image as mpimg
import matplotlib.markers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections

plt.ioff()


from titlebox import filename_table, filt_params_table, eps_table, i_table
from titlebox import movie_params_table
from utilities import remove_old_frames, frames_to_movie, clear_temp_files, \
	print_still
from config import gnuplot_str


def complexes_coords(filtration):
	"""Replaces each landmark_ID with corresponding coordinates"""
	coords_array = []
	for row in filtration.complexes:
		new_row = []
		for simplex in row:
			simplex_coords = []
			for landmark_ID in simplex:
				landmark_coords = filtration.landmark_coords[landmark_ID]
				simplex_coords.append(landmark_coords)
			new_row.append(simplex_coords)
		coords_array.append(new_row)
	return np.asarray(coords_array)

def simplex_color(scheme, past_birth_time, birth_time, max_birth_time):
	if scheme is None:
		facecolor = 'C0'
		edgecolor = 'black'

	elif scheme == 'highlight new':
		if past_birth_time == birth_time:
			facecolor = 'C3'
			# edgecolor = 'C6'
			edgecolor = 'maroon'
		else:
			facecolor = 'C0'
			edgecolor = 'black'

	elif hasattr(scheme, '__len__') and scheme[0] == 'birth time gradient':
		cycles = scheme[1]
		prog = (past_birth_time / float(max_birth_time))
		c = divmod((prog * cycles), 1)[1] 	# modulo 1
		facecolor = (1, c, 1 - c)
		edgecolor = (.5, c, 1 - c)

	else:
		print 'ERROR:', scheme, 'is not a valid color scheme'
		sys.exit()

	return facecolor, edgecolor


def plot_witnesses_2D(subplot, attractor_data):
	attractor_data = np.array(attractor_data)
	x = attractor_data[:, 0]
	y = attractor_data[:, 1]
	subplot.scatter(
		x, y,
		color='black',
		marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'),
		facecolor='black',
		s=.1)

	subplot.set_aspect('equal')


def plot_landmarks_2D(subplot, landmark_data):
	landmark_data = np.array(landmark_data)
	x = landmark_data[:, 0]
	y = landmark_data[:, 1]
	# subplot.scatter(x, y, color='lime', s=35)
	subplot.scatter(x, y, color='springgreen', s=35)



def plot_complex_2D(subplot, filtration, i, color_scheme=None, alpha=1):

	complex_data = complexes_coords(filtration)

	for j, simplices_coords in enumerate(complex_data[:i + 1]):

		f_color, e_color = simplex_color(
			color_scheme, j, i, len(complex_data)
		)

		simplexes = collections.PolyCollection(
			simplices_coords,
			edgecolors=e_color,
			facecolors=f_color,
			lw=1.5,
			alpha=alpha,
			zorder=0,
			animated=True,
			antialiased=True)

		subplot.add_collection(simplexes)


def plot_witnesses_3D_mayavi():
	pass


def plot_landmarks_3D_mayavi():
	pass


def plot_complex_3D_mayavi():
	pass


def plot_all_3D_gnuplot(subplot, filtration, i, camera_angle):
	def add_arrow(simplex, cmds):
		set_arrow = ' '.join([
			'set arrow from',
			'{}, {}, {} to'.format(*simplex[0]),
			'{}, {}, {}'.format(*simplex[1]),
			# 'nohead lc "red"'
			'nohead lw 1'
		])
		cmds.append(set_arrow)


	def add_poly(simplex, cmds, poly_count):
		set_poly = '\n'.join([
			'set object {} polygon from \\'.format(poly_count),
			'{}, {}, {} to \\'.format(*simplex[0]),
			'{}, {}, {} to \\'.format(*simplex[1]),
			'{}, {}, {} to \\'.format(*simplex[2]),
			'{}, {}, {}'.format(*simplex[0]),
		])

		style_poly = ' '.join([
			'set object {} fc rgb "#1F77B4"'.format(poly_count),
			'fillstyle solid',
			'lw 1'
		])

		cmds.append(set_poly)
		cmds.append(style_poly)


	def write_gnuplot_script():
		witness_data = filtration.witness_coords
		landmark_data = filtration.landmark_coords
		complex_data = complexes_coords(filtration)

		np.savetxt('phomology/temp/witnesses.txt', witness_data)
		np.savetxt('phomology/temp/landmarks.txt', landmark_data)


		cmds = [
			'set terminal pngcairo size 800, 800',
			'set view {}, {}'.format(*camera_angle),
			# 'set output "phomology/frames/frame{:02d}.png"'.format(i),
			# 'set size ratio - 1',
			# 'unset border',
			# 'unset tics',
		]

		# plot complex
		complex_data = complex_data[:i + 1]
		poly_count = 1
		for complex in complex_data:
			for simplex in complex:
				if len(simplex) == 1:
					pass
				elif len(simplex) == 2:
					add_arrow(simplex, cmds)
				else:
					add_poly(simplex, cmds, poly_count)
					poly_count += 1


		# plot witnesses and landmarks
		wits_arg = '"phomology/temp/witnesses.txt" with points pt 7 ps .1 ' \
				   'lc "black" notitle'
		lands_arg = '"phomology/temp/landmarks.txt" with points pt 7 ps 1 notitle'
		lands_arg = '"phomology/temp/landmarks.txt" with points pt 7 ps 1 ' \
		            'lc rgb "#00FF7F" notitle'
		cmds.append('splot {}, {}'.format(wits_arg, lands_arg))

		cmds.append('q')

		with open('phomology/temp/gnuplot_cmds.txt', 'w') as f:
			f.write('\n'.join(cmds))

	write_gnuplot_script()

	try:
		p = subprocess.Popen([gnuplot_str, 'phomology/temp/gnuplot_cmds.txt'],
						 stdout=subprocess.PIPE)
	except OSError:
		print '''ERROR: Unable to open gnuplot. Ensure that 'gnuplot_str' in 
			  config.py is set to the appropriate command to launch gnuplot
			  on your system.'''
		sys.exit()

	out, err = p.communicate()
	f = io.BytesIO(out)

	try:
		img = mpimg.imread(f, format='png')
	except ValueError as e:
		print e
		print '''ERROR: Invalid PNG header. Ensure that you are using a recent
			  version of gnuplot (5+) and 'gnuplot_str' in config.py is the 
			  appropriate command to launch this version.'''
		sys.exit()

	subplot.axis('off')
	subplot.imshow(img)


def plot_complex(
		filt, i, out_filename,
		camera_angle=(70, 45), alpha=1, dpi=200
):
	fig = plt.figure(figsize=(9, 6), tight_layout=False, dpi=dpi)

	fname_ax =         plt.subplot2grid((12, 8), (0, 0), rowspan=1,  colspan=2)
	epsilon_ax =       plt.subplot2grid((12, 8), (2, 0), rowspan=1,  colspan=2)
	i_ax =             plt.subplot2grid((12, 8), (3, 0), rowspan=1,  colspan=2)
	filt_params_ax =   plt.subplot2grid((12, 8), (5, 0), rowspan=7,  colspan=2)
	plot_ax =          plt.subplot2grid((12, 8), (0, 3), rowspan=12, colspan=6)

	fig.subplots_adjust(left=.05, right=.95, bottom=.1, top=.9)
	filename_table(fname_ax, filt.name)
	filt_params_table(filt_params_ax, filt.params)

	eps_table(epsilon_ax, filt.epsilons[i])
	i_table(i_ax, i)

	witness_data = filt.witness_coords
	landmark_data = filt.landmark_coords
	amb_dim = filt.ambient_dim

	print 'plotting complex...'
	if amb_dim == 2:
		plot_witnesses_2D(plot_ax, witness_data)
		plot_landmarks_2D(plot_ax, landmark_data)
		plot_complex_2D(plot_ax, filt, i, alpha=alpha)
	else:
		plot_all_3D_gnuplot(plot_ax, filt, i, camera_angle)
	plt.savefig(out_filename)
	plt.close(fig)

def build_movie(
		filt,
		out_filename,
		color_scheme=None,
		camera_angle=(70, 45),
		alpha=1,
		dpi=200,

):


	print 'building movie...'
	remove_old_frames('phomology/frames/')
	fig = plt.figure(figsize=(9, 6), tight_layout=False, dpi=dpi)

	fname_ax =         plt.subplot2grid((12, 8), (0, 0), rowspan=1,  colspan=2)
	epsilon_ax =       plt.subplot2grid((12, 8), (2, 0), rowspan=1,  colspan=2)
	movie_params_ax =  plt.subplot2grid((12, 8), (4, 0), rowspan=1,  colspan=2)
	filt_params_ax =   plt.subplot2grid((12, 8), (5, 0), rowspan=7,  colspan=2)
	plot_ax =          plt.subplot2grid((12, 8), (0, 3), rowspan=12, colspan=6)

	fig.subplots_adjust(left=.05, right=.95, bottom=.1, top=.9)
	filename_table(fname_ax, filt.name)
	movie_params_table(movie_params_ax, (color_scheme, alpha, '2D'))
	filt_params_table(filt_params_ax, filt.params)

	witness_data = filt.witness_coords
	landmark_data = filt.landmark_coords
	amb_dim = filt.ambient_dim


	for i, eps in enumerate(filt.epsilons):
		print_still('\rplotting frame {} of {}'.format(i + 1, filt.num_div))

		if amb_dim == 2:
			plot_witnesses_2D(plot_ax, witness_data)
			plot_landmarks_2D(plot_ax, landmark_data)
			plot_complex_2D(plot_ax, filt, i, color_scheme, alpha)
		else:
			plot_all_3D_gnuplot(plot_ax, filt, i, camera_angle)
		eps_table(epsilon_ax, eps)
		plt.savefig('phomology/frames/frame%03d.png' % i)
		plot_ax.clear()


	plt.close(fig)
	print ''
	frames_to_movie(out_filename, 'phomology/frames/frame%03d.png')
	clear_temp_files('phomology/temp/')

