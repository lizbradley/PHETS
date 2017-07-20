import os, sys, subprocess, time, io
import numpy as np
from matplotlib import collections
import matplotlib.pyplot as pyplot
import matplotlib.markers

from matplotlib import animation
import matplotlib.image as mpimg


import Utilities
from TitleBox import add_filename_table, add_filt_params_table, update_epsilon, add_movie_params_table
from config import gnuplot_str


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


def make_frames(filtration, color_scheme, alpha, save_frames):

	def plot_2D_init(subplot, attractor_data, landmark_data):

		def plot_witnesses(subplot, attractor_data):
			attractor_data = np.array(attractor_data)
			x = attractor_data[:, 0]
			y = attractor_data[:, 1]
			return subplot.scatter(x, y, color='black', marker=matplotlib.markers.MarkerStyle(marker='o', fillstyle='full'), facecolor='black', s=.1)


		def plot_landmarks(subplot, landmark_data):
			landmark_data = np.array(landmark_data)
			x = landmark_data[:, 0]
			y = landmark_data[:, 1]
			return subplot.scatter(x, y, color='darkblue', s=35)

		subplot.set_aspect('equal')
		return [plot_witnesses(subplot, attractor_data), plot_landmarks(subplot, landmark_data)]


	def plot_2D_update(subplot, filtration, i):

		def plot_complex(subplot, i, complex_data):
			"""plots all complexes for full filtration"""
			patches = []

			triangle_count = 0
			for j, simplexes_coords in enumerate(complex_data[:i]):

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
				triangle_count += len(simplexes_coords)

			with open('output/run_info/num_triangles.txt', 'a') as f:
				f.write('frame {}: {}\n'.format(i, triangle_count))

			return patches

		return plot_complex(subplot, i, filtration.get_complex_plot_data())


	def plot_2D_update_gnuplot(subplot, filtration, i):

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
			complex_data = filtration.get_complex_plot_data()

			np.savetxt('witnesses.txt', witness_data)
			np.savetxt('landmarks.txt', landmark_data)

			complex_data = complex_data[:i]

			cmds = ['set terminal pngcairo size 500, 500',
					# 'set output "PH/frames/frame{:02d}.png"'.format(i),
					# 'set size ratio - 1',
					# 'unset border',
					# 'unset tics'
					]

			triangle_count = 1
			for complex in complex_data:
				for simplex in complex:
					if len(simplex) == 1:
						# print 'length 1 simplex ({}) encountered. skipping'.format(simplex)
						pass
					elif len(simplex) == 2:
						add_arrow(simplex, cmds)
					else:
						add_poly(simplex, cmds, triangle_count)
						triangle_count += 1

			with open('output/run_info/num_triangles.txt', 'a') as f:
				f.write('frame {}: {}\n'.format(i, triangle_count))

			# plot witnesses and landmarks
			cmds.append('''plot \
						"witnesses.txt" with points pt 7 ps .1 lc "black" notitle, \
						"landmarks.txt" with points pt 7 ps 1 notitle''')

			cmds.append('q')


			with open('PH/temp_data/gnuplot_cmds.txt', 'w') as f:
				f.write('\n'.join(cmds))


		write_gnup_script()
		p = subprocess.Popen([gnuplot_str, 'PH/temp_data/gnuplot_cmds.txt'], stdout=subprocess.PIPE)

		out, err = p.communicate()
		f = io.BytesIO(out)
		img = mpimg.imread(f, format='png')


		subplot.axis('off')
		return subplot.imshow(img),


	def plot_3D_update(subplot, filtration, i):
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

			np.savetxt('witnesses.txt', witness_data)
			np.savetxt('landmarks.txt', landmark_data)

			complex_data = complex_data[:i]

			cmds = ['set terminal pngcairo size 700, 700',
					# 'set output "PH/frames/frame{:02d}.png"'.format(i),
					# 'set size ratio - 1',
					# 'unset border',
					# 'unset tics'
					]

			triangle_count = 1
			for complex in complex_data:
				for simplex in complex:
					if len(simplex) == 1:
						# print 'length 1 simplex ({}) encountered. skipping'.format(simplex)
						pass
					elif len(simplex) == 2:
						add_arrow(simplex, cmds)
					else:
						add_poly(simplex, cmds, triangle_count)
						triangle_count += 1

			with open('output/run_info/num_triangles.txt', 'a') as f:
				f.write('frame {}: {}\n'.format(i, triangle_count))

			# plot witnesses and landmarks
			cmds.append('''splot \
							"witnesses.txt" with points pt 7 ps .1 lc "black" notitle, \
							"landmarks.txt" with points pt 7 ps 1 notitle''')

			cmds.append('q')

			with open('PH/temp_data/gnuplot_cmds.txt', 'w') as f:
				f.write('\n'.join(cmds))

		write_gnup_script()
		p = subprocess.Popen(['gnuplot-x11', 'PH/temp_data/gnuplot_cmds.txt'], stdout=subprocess.PIPE)

		out, err = p.communicate()
		f = io.BytesIO(out)
		img = mpimg.imread(f, format='png')

		# debugging
		# if i ==3:
		# 	fig = pyplot.figure(figsize=(6, 6), tight_layout=True, dpi=300)
		# 	ax = fig.add_subplot(111)
		# 	ax.axis('off')
		# 	ax.imshow(img, interpolation='none')
		# 	fig.savefig('test.png')
		# 	sys.exit()

		subplot.axis('off')
		return subplot.imshow(img),




	fname_ax = 			pyplot.subplot2grid((12, 8), (0, 0), rowspan=2, colspan=2)
	epsilon_ax = 		pyplot.subplot2grid((12, 8), (2, 0), rowspan=2, colspan=2)
	movie_params_ax =	pyplot.subplot2grid((12, 8), (4, 0), rowspan=2, colspan=2)
	filt_params_ax =	pyplot.subplot2grid((12, 8), (6, 0), rowspan=6, colspan=2)
	plot_ax = 			pyplot.subplot2grid((12, 8), (0, 2), rowspan=12, colspan=6)

	add_filename_table(fname_ax, filtration.filename)
	add_movie_params_table(movie_params_ax, (color_scheme, alpha, '2D'))
	add_filt_params_table(filt_params_ax, filtration.params)

	witness_data = filtration.witness_coords
	landmark_data = filtration.landmark_coords

	amb_dim = filtration.ambient_dim
	if amb_dim not in (2, 3):
		print 'ERROR: invalid ambient dimension {}, must be 2 or 3'.format(amb_dim)
		sys.exit()

	with open('output/run_info/num_triangles.txt', 'a') as f:
		f.truncate(0)


	def init():
		if amb_dim == 2:
			return plot_2D_init(plot_ax, witness_data, landmark_data)
		else:
			return plot_ax.plot([])		# FuncAnimation wants an artist object

	def animate(i):
		sys.stdout.write('\rplotting frame {} of {}'.format(i, filtration.num_div))
		sys.stdout.flush()

		if amb_dim == 2:
			comp_plot = plot_2D_update(plot_ax, filtration, i)
			# comp_plot = plot_2D_update_gnuplot(plot_ax, filtration, i)
		else:
			comp_plot = plot_3D_update(plot_ax, filtration, i)

		eps = update_epsilon(epsilon_ax, i, filtration.epsilons)

		if save_frames: pyplot.savefig('frames/image%03d.png' % i)

		return list(comp_plot) + list(eps)

	return init, animate




def make_movie(
		filtration,
		out_filename,
		color_scheme='none',		  	# as of now, 'none', 'highlight new', or 'birth_time gradient'
		camera_angle=(135, 55),  		# for 3D mode. [azimuthal, elevation]
		alpha=1, 					 	# opacity (float, 0...1 : transparent...opaque)
		dpi=150,  						# dots per inch (resolution)
		hide_1simplexes=False,			# i need to find a way to optimize the plotting of 1-simplexes(lines) 3D plotting, as of now they slow mayavi significantly.
		save_frames=False,  			# save frames to /frames/ dir
		framerate=1						# number of frames per second. for a constant max_frames, higher framerate will make a shorter movie.

):
	start = time.time()

	movie_info = [color_scheme, camera_angle, alpha]
	fnames = [filtration.filename, out_filename]

	fig = pyplot.figure(figsize=(9, 6), tight_layout=True, dpi=dpi)

	print 'building movie...'
	init, animate = make_frames(filtration, color_scheme, alpha, save_frames=save_frames)
	ani = animation.FuncAnimation(fig, animate, init_func=init, frames=filtration.num_div + 1,
								  blit=True, repeat=False)

	# FuncAnimation.save() uses pipes to send frames to ffmpeg, which is significantly faster than saving to png.
	# However the videos it creates do not work well if fps is low (~ 1) because it uses fps for the output framerate.
	# As a workaround, ani.save(fps=10) is used and then ffmpeg is called to reduce the speed of the video by a 10x
	# Another option would be to stream to FFMPEG ourselves: http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/


	ani.save('output/PH/temp.mp4', fps=10)

	print '... done.'

	# correct framerate
	subprocess.call(['ffmpeg',
					 '-loglevel', 'panic', '-y',
					 '-i', 'output/PH/temp.mp4',
					 '-filter:v', 'setpts={:d}*PTS'.format(int(10 / framerate)),
					 out_filename])

	os.remove('output/PH/temp.mp4')


	print 'time elapsed: {} s'.format(time.time() - start)





if __name__ == '__main__':
	pass