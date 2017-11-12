import numpy as np
from matplotlib.colorbar import ColorbarBase
import numpy.ma as ma
from matplotlib.colors import from_levels_and_colors
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt; plt.ioff()
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from titlebox import filename_table, filt_params_table

# from Utilities import mem_profile
# f=open("output/run_info/group_by_birth_time_memory.txt","wb")
# f2=open("output/run_info/expand_to_2simplexes_memory.txt","wb")
# f3=open("output/run_info/build_perseus_in_file_memory.txt","wb")
# f4=open("output/run_info/make_figure_memory.txt","wb")

# @mem_profile(f, MEMORY_PROFILE_ON)
# @profile(stream=f)



def PD_ax(ax, cbar_ax, filtration):

	ax.set_aspect('equal')
	min_lim = 0
	max_lim = np.max(filtration.epsilons)
	ax.set_xlim(min_lim, max_lim)
	ax.set_ylim(min_lim, max_lim)
	ax.set_xlabel('birth ($\epsilon$)')
	ax.set_ylabel('death ($\epsilon$)')
	ax.grid(which=u'major', zorder=0)
	ax.minorticks_on()

	ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k')		# diagonal line

	data = filtration.PD()
	if data == 'empty':
		return

	sc = None
	if len(data.mortal) > 0:
		x_mor, y_mor, count_mor = data.mortal
		sc = ax.scatter(
			x_mor, y_mor, s=70,
			c=count_mor, alpha=.8,
			clip_on=True, zorder=100,
			vmin=1, vmax=5
		)
	if len(data.immortal) > 0:
		x_imm, count_imm = data.immortal
		y_imm = [max_lim for i in x_imm]
		sc = ax.scatter(
			x_imm, y_imm, marker='^', s=120,
			c=count_imm, alpha=.8,
			clip_on=False, zorder=100,
			vmin=1, vmax=5
		)

	levels = [1, 2, 3, 4, 5]
	cb = plt.colorbar(
		sc,
		cax=cbar_ax,
		extend='max',
		extendrect=True,
		extendfrac=.2,
		values=levels
	)

	cb.ax.text(1.5, 0.10, '1')
	cb.ax.text(1.5, 0.35, '2')
	cb.ax.text(1.5, 0.60, '3')
	cb.ax.text(1.5, 0.85, '4')
	cb.ax.text(1.5, 1.10, '5+')


# @profile(stream=f4)
def PD(filt, out_filename):
	print 'plotting persistence diagram...'

	fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=100)

	fname_ax = 		plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	params_ax = 	plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		plt.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		plt.subplot2grid((6, 10), (0, 9), rowspan=6)

	pos = cbar_ax.get_position()
	cbar_ax.set_position(
		[pos.x0 + .1, pos.y0 - .05, pos.x1 - pos.x0 + -.05, pos.y1 - pos.y0 + .1]
	)

	PD_ax(plot_ax, cbar_ax, filt)
	filename_table(fname_ax, filt.filename)
	filt_params_table(params_ax, filt.params)

	plt.savefig(out_filename)
	plt.close(fig)


def plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=False):

	def annotate():
		offset = (1.41 / (len(x) - 1)) / 2
		for i, x_ in enumerate(x):
			for j, y_ in enumerate(y):
				plot_ax.text(
					x_ + offset, y_ + offset, '%.3f' % z[j, i],
					horizontalalignment='center',
					verticalalignment='center',
					color='salmon'
				)

	def extend_domain(x, y):
		"""
		from pcolormesh() documentation:
			Ideally, the dimensions of X and Y should be one greater than
			those of C; if the dimensions are the same, then the last row and
			column of C will be ignored.
		"""
		d = x[1] - x[0]
		x = np.append(x, x[-1] + d)
		y = np.append(y, y[-1] + d)
		return x, y

	plot_ax.set_aspect('equal')
	viridis = get_cmap('viridis')
	colors = [viridis(i) for i in np.linspace(0, 1, 13)]
	levels = np.concatenate([[0, .0001], np.arange(1, 10), [50, 100]])
	cmap, norm = from_levels_and_colors(levels, colors, extend='max')
	zm = ma.masked_where(np.isnan(z), z)

	if x is not None and y is not None:
		x, y = extend_domain(x, y)
		plot_ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, clip_on=False)
		if annot: annotate()
	elif x is None and y is None:
		plot_ax.pcolormesh(zm, cmap=cmap, norm=norm, clip_on=False)
	else:
		print 'ERROR: plot_heatmap: x and y must both be None or array-like'
		sys.exit()

	ColorbarBase(cbar_ax, norm=norm, cmap=cmap, ticks=levels, extend='max')
	return cmap


def PRF_ax(filtration, ax, cbar_ax=None, annot_hm=False):

	if cbar_ax is None:
		divider = make_axes_locatable(ax)
		cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

	if isinstance(filtration, filtration.Filtration):
		z = filtration._PRF()
		x = y = filtration.epsilons
		plot_heatmap(ax, cbar_ax, x, y, z, annot_hm)
	else:   # 2d array
		z = filtration
		plot_heatmap(ax, cbar_ax, None, None, z, annot_hm)



def PRF(filtration, out_filename, annot_hm=False):
	print "plotting PRF..."

	fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=100)
	fname_ax = 		plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	params_ax = 	plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		plt.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		plt.subplot2grid((6, 10), (0, 9), rowspan=6)

	######## from here ##########

	func = filtration.PRF()
	in_filename = filtration.filename
	params = filtration.params

	x, y, z, max_lim = func

	if len(x.shape) == 2: 			# meshgrid format
		x, y = x[0], y[:, 0]		# reduce to arange format


	plot_heatmap(plot_ax, cbar_ax, x, y, z, annot=annot_hm)

	####### to here ###########
	# should eventually be replaced by PRF_ax

	filename_table(fname_ax, in_filename)
	filt_params_table(params_ax, params)


	fig.savefig(out_filename)
	plt.close(fig)


