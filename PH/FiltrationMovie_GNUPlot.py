import os
import sys
import subprocess
import numpy as np

import Utilities
from TitleBox import add_filename_table, add_filt_params_table, update_epsilon, add_movie_params_table

from FiltrationMovie import remove_old_frames, get_simplex_color


import sys
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt



def plot_frame_2D(filtration, i):

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

	print 'frame ' + str(i)

	witness_data = filtration.witness_coords
	landmark_data = filtration.landmark_coords
	complex_data = filtration.get_complex_plot_data()

	np.savetxt('witnesses.txt', witness_data)
	np.savetxt('landmarks.txt', landmark_data)


	complex_data = complex_data[:i]

	cmds = ['set terminal pngcairo size 500, 500',
			'set output "PH/frames/frame{:02d}.png"'.format(i),
			# 'set size ratio - 1',
			# 'unset border',
			# 'unset tics'
			]



	poly_count = 1
	for complex in complex_data:
		for simplex in complex:
			if len(simplex) == 2:
				add_arrow(simplex, cmds)
			else:
				add_poly(simplex, cmds, poly_count)
				poly_count += 1
			

	print 'number of polygons: ', poly_count
	# plot witnesses and landmarks
	cmds.append('''plot \
				"witnesses.txt" with points pt 7 ps .1 lc "black" notitle, \
				"landmarks.txt" with points pt 7 ps 1 notitle''')

	cmds.append('q')


	import matplotlib.image as mpimg
	import os

	with open('gnuplot_cmds.txt', 'w') as f:
		f.write('\n'.join(cmds))

	p = subprocess.Popen(['gnuplot-x11', 'gnuplot_cmds.txt'], stdout=subprocess.PIPE)
	# subprocess.call(['gnuplot-x11', 'gnuplot_cmds.txt'])
	# os.system('gnuplot-x11 gnuplot_cmds.txt')

	import io

	# for 3D movies, stream to matplotlib rather than writing to file


	# out, err = p.communicate()
	# f = io.BytesIO(out)
	# img = mpimg.imread(f, format='png')
	#
	# print 'saving...'
	#
	# fig = plt.figure(figsize=(6, 6), tight_layout=True, dpi=300)
	# ax = fig.add_subplot(111)
	# ax.axis('off')
	# ax.imshow(img, interpolation='none')
	#
	# fig.savefig('test.png')
	#
	#
	#

def make_frame(filtration, i):

	# matplotlib stuff

	plot_frame_2D(filtration, i)



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
	for i in xrange(filtration.num_div):
		make_frame(filtration, i)

