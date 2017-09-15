import io, subprocess, sys, os

import matplotlib.image as mpimg
import matplotlib.markers
import matplotlib.pyplot as pyplot
import numpy as np
from matplotlib import collections

from TitleBox import add_filename_table, add_filt_params_table, update_epsilon, add_movie_params_table
from Utilities import remove_old_frames, frames_to_movie, clear_temp_files
from config import gnuplot_str


def get_simplex_color(scheme, past_birth_time, present_birth_time, max_birth_time):
	"""helper for plot_complex()"""
	if scheme == 'none':
		facecolor = 'lightblue'
		edgecolor = 'black'

	elif scheme == 'highlight new':
		if past_birth_time == present_birth_time:
			facecolor = 'red'
			edgecolor = 'firebrick'
		else:
			facecolor = 'lightblue'
			edgecolor = 'black'

	elif hasattr(scheme, '__len__') and scheme[0] == 'birth time gradient':
		cycles = scheme[1]
		prog = divmod(((past_birth_time / float(max_birth_time)) * cycles), 1)[1] 	# modulo 1
		facecolor = (1, prog, 1 - prog)
		edgecolor = (.5, prog, 1 - prog)

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


def plot_landmarks_2D(subplot, landmark_data):
	landmark_data = np.array(landmark_data)
	x = landmark_data[:, 0]
	y = landmark_data[:, 1]
	subplot.scatter(x, y, color='darkblue', s=35)



def plot_complex_2D(subplot, filtration, i, color_scheme, alpha):

	complex_data = filtration.get_complex_plot_data()


	for j, simplexes_coords in enumerate(complex_data[:i + 1]):

		f_color, e_color = get_simplex_color(color_scheme, j, i, len(complex_data))

		simplexes = collections.PolyCollection(
			simplexes_coords,
			edgecolors=e_color,
			facecolors=f_color,
			lw=1,
			alpha=alpha,
			zorder=0,
			animated=True,
			antialiased=True)

		subplot.add_collection(simplexes)




def plot_all_3D(subplot, filtration, i, camera_angle):
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
			'set object {} fc rgb "#999999"'.format(poly_count),
			'fillstyle solid',
			'lw 1'
		])

		cmds.append(set_poly)
		cmds.append(style_poly)


	def write_gnup_script():
		witness_data = filtration.witness_coords
		landmark_data = filtration.landmark_coords
		complex_data = filtration.get_complex_plot_data()

		np.savetxt('PH/temp/witnesses.txt', witness_data)
		np.savetxt('PH/temp/landmarks.txt', landmark_data)


		cmds = ['set terminal pngcairo size 700, 700',
				'set view {}, {}'.format(*camera_angle),
				# 'set output "PH/frames/frame{:02d}.png"'.format(i),
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
		wits_arg = '"PH/temp/witnesses.txt" with points pt 7 ps .1 lc "black" notitle'
		lands_arg = '"PH/temp/landmarks.txt" with points pt 7 ps 1 notitle'
		cmds.append('splot {}, {}'.format(wits_arg, lands_arg))



		cmds.append('q')

		with open('PH/temp/gnuplot_cmds.txt', 'w') as f:
			f.write('\n'.join(cmds))




	write_gnup_script()
	p = subprocess.Popen([gnuplot_str, 'PH/temp/gnuplot_cmds.txt'], stdout=subprocess.PIPE)

	out, err = p.communicate()
	f = io.BytesIO(out)
	img = mpimg.imread(f, format='png')

	subplot.axis('off')
	subplot.imshow(img)




def plot_all_2D_gnuplot(subplot, filtration, i):

	def add_arrow(simplex, cmds):
		set_arrow = ' '.join([
			'set arrow from',
			'{}, {} to'.format(*simplex[0]),
			'{}, {}'.format(*simplex[1]),
			# 'nohead lc "red"'
			'nohead lw 1'
		])
		cmds.append(set_arrow)

	def add_poly(simplex, cmds, poly_count):
		set_poly = '\n'.join([
			'set object {} polygon from \\'.format(poly_count),
			'{}, {} to \\'.format(*simplex[0]),
			'{}, {} to \\'.format(*simplex[1]),
			'{}, {} to \\'.format(*simplex[2]),
			'{}, {}'.format(*simplex[0]),
		])

		style_poly = ' '.join([
			'set object {} fc rgb "#999999"'.format(poly_count),
			'fillstyle solid',
			'lw 1'
		])

		cmds.append(set_poly)
		cmds.append(style_poly)



	def write_gnup_script():
		witness_data = filtration.witness_coords
		landmark_data = filtration.landmark_coords
		complex_data = filtration.get_complex_plot_data()[:i + 1]

		np.savetxt('PH/temp/witnesses.txt', witness_data)
		np.savetxt('PH/temp/andmarks.txt', landmark_data)

		cmds = ['set terminal pngcairo size 700, 700',
				# 'set output "PH/frames/frame{:02d}.png"'.format(i),
				# 'set size ratio - 1',
				# 'unset border',
				# 'unset tics'
				]




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
		cmds.append('q')

		# plot witnesses and landmarks
		wits_arg = '"PH/temp/witnesses.txt" with points pt 7 ps .1 lc "black" notitle'
		lands_arg = '"PH/temp/landmarks.txt" with points pt 7 ps 1 notitle'
		cmds.append('plot {}, {}'.format(wits_arg, lands_arg))




		with open('PH/temp/gnuplot_cmds.txt', 'w') as f:
			f.write('\n'.join(cmds))


	write_gnup_script()
	p = subprocess.Popen([gnuplot_str, 'PH/temp/gnuplot_cmds.txt'], stdout=subprocess.PIPE)

	out, err = p.communicate()
	f = io.BytesIO(out)
	img = mpimg.imread(f, format='png')


	subplot.axis('off')


	subplot.imshow(img),




def make_movie(
		filt,
		out_filename,
		color_scheme='none',		  	# as of now, 'none', 'highlight new', or 'birth_time gradient'
		camera_angle=(70, 45),  		# for 3D mode. [azimuthal, elevation]
		alpha=1, 					 	# opacity (float, 0...1 : transparent...opaque)
		dpi=200,  						# dots per inch (resolution)

):
	print 'building movie...'
	remove_old_frames('PH/frames/')
	fig = pyplot.figure(figsize=(9, 6), tight_layout=True, dpi=dpi)

	fname_ax = pyplot.subplot2grid((12, 8), (0, 0), rowspan=2, colspan=2)
	epsilon_ax = pyplot.subplot2grid((12, 8), (2, 0), rowspan=2, colspan=2)
	movie_params_ax = pyplot.subplot2grid((12, 8), (4, 0), rowspan=2, colspan=2)
	filt_params_ax = pyplot.subplot2grid((12, 8), (6, 0), rowspan=6, colspan=2)
	plot_ax = pyplot.subplot2grid((12, 8), (0, 2), rowspan=12, colspan=6)

	add_filename_table(fname_ax, filt.filename)
	add_movie_params_table(movie_params_ax, (color_scheme, alpha, '2D'))
	add_filt_params_table(filt_params_ax, filt.params)

	witness_data = filt.witness_coords
	landmark_data = filt.landmark_coords

	amb_dim = filt.ambient_dim
	if amb_dim not in (2, 3):
		print 'ERROR: invalid ambient dimension {}, must be 2 or 3'.format(amb_dim)
		sys.exit()

	for i, eps in enumerate(filt.epsilons):
		sys.stdout.write('\rplotting frame {} of {}'.format(i + 1, filt.num_div))
		sys.stdout.flush()

		if amb_dim == 2:
			plot_witnesses_2D(plot_ax, witness_data)
			plot_landmarks_2D(plot_ax, landmark_data)
			plot_complex_2D(plot_ax, filt, i, color_scheme, alpha)
			# plot_all_2D_gnuplot(plot_ax, filt, i)		# to test consistency
		else:
			plot_all_3D(plot_ax, filt, i, camera_angle)	# uses gnuplot

		update_epsilon(epsilon_ax, eps)

		pyplot.savefig('PH/frames/frame%03d.png' % i)
		plot_ax.clear()


	pyplot.close(fig)
	print ''
	frames_to_movie(out_filename, 'PH/frames/frame%03d.png')
	clear_temp_files('PH/temp/')







if __name__ == '__main__':
	pass