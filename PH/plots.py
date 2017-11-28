import sys

import numpy as np
import numpy.ma as ma

from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import from_levels_and_colors
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt; plt.ioff()
from mpl_toolkits.axes_grid1 import make_axes_locatable

from titlebox import filename_table, filt_params_table


def colorbar_ax(cbar_ax, levels):
	viridis = get_cmap('viridis')
	colors = [viridis(i) for i in np.linspace(0, 1, len(levels))]
	cmap, norm = from_levels_and_colors(levels, colors, extend='max')

	cbar = ColorbarBase(
		cbar_ax,
		norm=norm,
		cmap=cmap,
		ticks=levels,
		extend='max',
		extendrect=True,
	)
	return cmap, norm


def PD_ax(ax, cbar_ax, pd):

	ax.set_aspect('equal')
	min_lim = 0
	max_lim = pd.lim
	ax.set_xlim(min_lim, max_lim)
	ax.set_ylim(min_lim, max_lim)
	ax.set_xlabel('birth ($\epsilon$)')
	ax.set_ylabel('death ($\epsilon$)')
	ax.grid(which=u'major', zorder=0)
	ax.minorticks_on()
	ax.ticklabel_format(axis='both', style='sci',  scilimits=(0,0))

	ax.plot([min_lim, max_lim], [min_lim, max_lim], color='k')  # diagonal line

	levels = [1, 2, 3, 4, 5]
	cmap, norm = colorbar_ax(cbar_ax, levels)

	x_mor, y_mor, count_mor = pd.mortal
	ax.scatter(
		x_mor, y_mor, s=70,
		c=count_mor, alpha=.8,
		clip_on=True, zorder=100,
		cmap=cmap, norm=norm
	)

	x_imm, count_imm = pd.immortal
	y_imm = [max_lim for i in x_imm]
	ax.scatter(
		x_imm, y_imm, marker='^', s=120,
		c=count_imm, alpha=.8,
		clip_on=False, zorder=100,
		cmap=cmap, norm=norm
	)



def PD_fig(filt, out_filename):
	print 'plotting persistence diagram...'

	fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=100)

	fname_ax = 		plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	params_ax = 	plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		plt.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		plt.subplot2grid((6, 10), (0, 9), rowspan=6)

	PD_ax(plot_ax, cbar_ax, filt.PD)
	filename_table(fname_ax, filt.name)
	filt_params_table(params_ax, filt.params)

	plt.savefig(out_filename)
	plt.close(fig)


def heatmap_ax(plot_ax, cbar_ax, z, dom=None, annot=False):

	def annotate():
		offset = (1.41 / (len(x) - 1)) / 2
		for i, x_ in enumerate(x):
			for j, y_ in enumerate(y):
				plot_ax.text(
					x_ + offset, y_ + offset, '%.3f' % z[j, i],
					ha='center',
					va='center',
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

	plot_ax.ticklabel_format(axis='both', style='sci',  scilimits=(0, 0))
	plot_ax.set_aspect('equal')
	levels = np.concatenate([[0, .0001], np.arange(1, 10), [50, 100]])
	cmap, norm = colorbar_ax(cbar_ax, levels)
	zm = ma.masked_where(np.isnan(z), z)

	if dom is None:
		plot_ax.pcolormesh(zm, cmap=cmap, norm=norm, clip_on=False)
	else:
		x, y = extend_domain(dom, dom)
		plot_ax.pcolormesh(x, y, zm, cmap=cmap, norm=norm, clip_on=False)
		if annot: annotate()

	return cmap


def PRF_ax(prf, ax, cbar_ax=None, annot_hm=False):

	ax.set_xlabel('birth ($\epsilon$)')
	ax.set_ylabel('death ($\epsilon$)')

	if cbar_ax is None:
		divider = make_axes_locatable(ax)
		cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

	from filtration import PRankFunction
	if isinstance(prf, PRankFunction):
		z = prf.data
		heatmap_ax(ax, cbar_ax, z, dom=prf.epsilons, annot=annot_hm)
	else:   # 2d array
		z = prf
		heatmap_ax(ax, cbar_ax, z, annot=annot_hm)



def PRF_fig(filt, out_filename, annot_hm=False):
	print "plotting PRF..."

	fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=100)
	fname_ax = 		plt.subplot2grid((6, 10), (0, 0), rowspan=1, colspan=3)
	params_ax = 	plt.subplot2grid((6, 10), (2, 0), rowspan=4, colspan=3)
	plot_ax = 		plt.subplot2grid((6, 10), (0, 3), rowspan=6, colspan=6)
	cbar_ax = 		plt.subplot2grid((6, 10), (0, 9), rowspan=6)

	PRF_ax(filt.prf, plot_ax, cbar_ax, annot_hm)

	filename_table(fname_ax, filt.name)
	filt_params_table(params_ax, filt.params)

	fig.savefig(out_filename)
	plt.close(fig)


