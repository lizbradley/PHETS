import os
import sys
import subprocess
import numpy as np
from matplotlib import collections
import matplotlib.pyplot as pyplot
import matplotlib.markers

from matplotlib import animation

import Utilities
from TitleBox import add_filename_table, add_filt_params_table, update_epsilon, add_movie_params_table


def remove_old_frames():
	dir = 'PH/frames'
	for f in os.listdir(dir):
		if f.endswith(".png"):
			os.remove(dir + f)


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
		prog = divmod(((past_birth_time / float(max_birth_time)) * cycles), 1)[1] # modulo 1
		facecolor = (1, prog, 1 - prog)
		edgecolor = (.5, prog, 1 - prog)

	else:
		print 'error:', scheme, 'is not a valid color scheme'
	return facecolor, edgecolor


def make_frames_2D(filtration, color_scheme, alpha, save_frames):
	def plot_witnesses(subplot, witness_data):
		witness_data = np.array(witness_data)
		x = witness_data[:, 0]
		y = witness_data[:, 1]
		return subplot.scatter(x, y, color='black', marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'), facecolor='black', s=.1)

	def plot_landmarks(subplot, landmark_data):
		landmark_data = np.array(landmark_data)
		x = landmark_data[:, 0]
		y = landmark_data[:, 1]
		return subplot.scatter(x, y, color='darkblue', s=35)

	def plot_complex(subplot, i):
		"""plots all complexes for full filtration"""
		patches = []
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

			patches.append(subplot.add_collection(simplexes))

		return patches

	def plot_filtration_gnuplot(subplot, i):
		import Gnuplot

		def get_witnesses(witness_data):
			witness_data = np.array(witness_data)
			x = witness_data[:, 0]
			y = witness_data[:, 1]


			d = Gnuplot.Data(x, y,
							 with_='points pt 7 ps .5')
			return d

		def get_landmarks(landmark_data):
			landmark_data = np.array(landmark_data)
			x = landmark_data[:, 0]
			y = landmark_data[:, 1]

			d = Gnuplot.Data(x, y,
							 with_='points pt 7 ps 2')
			return d

		def get_complexes(complex_data):



			for j, simplexes_coords in enumerate(complex_data[:i + 1]):

				f_color, e_color = get_simplex_color(color_scheme, j, i, len(complex_data))

				g = Gnuplot.Gnuplot()

				g('set style fill transparent solid 0.6')

				simps = []

				for simplex_coords in simplexes_coords:

					d = Gnuplot.Data(simplex_coords,
								 with_="filledcurves closed lw 2 fc rgb 'black'")

					simps.append(d)

				return simps



		g = Gnuplot.Gnuplot()
		g('set style data points')

		w = get_witnesses(witness_data)
		l = get_landmarks(landmark_data)
		c = get_complexes(complex_data)

		g.plot(w, l, *c)
		# g.plot(c)
		g.hardcopy('gp_test.ps', enhanced=1, color=1)
		g.reset()
		print 'goodbye'
		sys.exit()





	fname_ax = 			pyplot.subplot2grid((12, 8), (0, 0), rowspan=2, colspan=2)
	epsilon_ax = 		pyplot.subplot2grid((12, 8), (2, 0), rowspan=2, colspan=2)
	movie_params_ax =	pyplot.subplot2grid((12, 8), (4, 0), rowspan=2, colspan=2)
	filt_params_ax =	pyplot.subplot2grid((12, 8), (6, 0), rowspan=6, colspan=2)
	plot_ax = 			pyplot.subplot2grid((12, 8), (0, 2), rowspan=12, colspan=6)

	add_filename_table(fname_ax, filtration.filename)

	add_movie_params_table(movie_params_ax, (color_scheme, alpha, '2D'))
	add_filt_params_table(filt_params_ax, filtration.params)


	# IDA paper formatting #
	# plot_ax.tick_params(labelsize=23)
	# plot_ax.xaxis.major.locator.set_params(nbins=5)
	# plot_ax.yaxis.major.locator.set_params(nbins=5)

	witness_data = filtration.witness_coords
	landmark_data = filtration.landmark_coords
	complex_data = filtration.get_complexes_mpl()

	plot_filtration_gnuplot(plot_ax, 10)

	def init():
		print 'initializing...'
		plot_ax.set_aspect('equal')
		witnesses = plot_witnesses(plot_ax, witness_data)
		landmarks = plot_landmarks(plot_ax, landmark_data)
		ret_list = [witnesses, landmarks]
		return ret_list

	def animate(i):
		print 'frame', i
		ret_comp = plot_complex(plot_ax, i)
		ret_eps = update_epsilon(epsilon_ax, i, filtration)
		ret_list = list(ret_comp)
		ret_list.extend(ret_eps)

		if save_frames: pyplot.savefig('frames/image%03d.png' % i)

		return ret_list

	return init, animate


def make_movie(
		filtration,
		out_filename,
		color_scheme='none',		  	# as of now, 'none', 'highlight new', or 'birth_time gradient'
		camera_angle=(135, 55),  		# for 3D mode. [azimuthal, elevation]
		alpha=1, 					 	# opacity (float, 0...1 : transparent...opaque)
		dpi=150,  						# dots per inch (resolution)
		max_frames=None,  				# cut off frame (for testing or when only interested in the beginning of a movie)
		hide_1simplexes=False,			# i need to find a way to optimize the plotting of 1-simplexes(lines) 3D plotting, as of now they slow mayavi significantly.
		save_frames=False,  			# save frames to /frames/ dir
		framerate=1						# number of frames per second. for a constant max_frames, higher framerate will make a shorter movie.

):

	# remove_old_frames()

	# Utilities.check_overwrite(out_filename)


	movie_info = [color_scheme, camera_angle, alpha]

	fnames = [filtration.filename, out_filename]


	fig = pyplot.figure(figsize=(9, 6), tight_layout=True, dpi=dpi)

	if filtration.ambient_dim == 2:
		print 'building movie...'
		init, animate = make_frames_2D(filtration, color_scheme, alpha, save_frames=save_frames)
		ani = animation.FuncAnimation(fig, animate, init_func=init, frames=filtration.num_div, blit=True, repeat=False)

		# FuncAnimation.save() uses pipes to send frames to ffmpeg, which is significantly faster than saving to png.
		# However the videos it creates do not work well if fps is low (~ 1) because it uses fps for the output framerate.
		# As a workaround, ani.save(fps=10) is used and then ffmpeg is called to reduce the speed of the video by a 10x

		# FuncAnimation.save() offers blitting, which should improve performance even further, but despite my best efforts
		# to configure it correctly, it doesn't make much any(?) difference in time-to-run. However, this may be a
		# consequence of all simplexes being hidden in the init function.

		print 'saving...'
		ani.save('output/PH/temp.mp4', fps=10)
		print 'correcting framerate...'

		subprocess.call(['ffmpeg', '-y', '-i',
						 'output/PH/temp.mp4',
						 '-filter:v', 'setpts={:d}*PTS'.format(int(10 / framerate)),
						 out_filename])

		os.remove('output/PH/temp.mp4')







if __name__ == '__main__':
	pass