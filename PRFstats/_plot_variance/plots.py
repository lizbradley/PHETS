import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PH.plots import heatmap_ax, PRF_colorbar_ax
from PH.titlebox import filenames_table, filt_params_table
from PRFstats.helpers import is_filt_param, is_weight_func
from utilities import clear_dir, make_dir


def weight_function_fig(f, num_div, fname):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	div = make_axes_locatable(ax)
	cax = div.append_axes('right', size='10%', pad=.2)

	x = y = np.linspace(0, 2 ** .5, num_div)
	xx, yy = np.meshgrid(x, y)
	z = f(xx, yy)
	if isinstance(z, int):
		z = xx * 0 + z

	mask = lambda x, y: x > y
	mask = mask(xx, yy)
	mask = np.where(mask is True, np.nan, 1)
	z = np.multiply(z, mask)

	heatmap_ax(ax, cax, x, y, z)
	plt.savefig('weight_functions/{}.png'.format(fname))


def heatmaps_figs(
		hmaps,
		vary_param_1,
        vary_param_2,
		legend_labels_1,
		legend_labels_2,
		out_dir,
		in_fname,
		annot_hm,
):
	out_dir = out_dir + '/heatmaps'
	make_dir(out_dir)
	if not clear_dir(out_dir):
		print 'skipping heatmaps'
		return

	print 'plotting heatmaps...'

	def make_hmap_fig(hmaps):

		fig = plt.figure(figsize=(12, 8))

		ax1 = fig.add_subplot(231)
		ax2 = fig.add_subplot(232)
		ax3 = fig.add_subplot(233)
		ax4 = fig.add_subplot(234)
		ax5 = fig.add_subplot(235)
		ax6 = fig.add_subplot(236)

		cax = fig.add_axes([.935, .1, .025, .78])

		import warnings
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			fig.tight_layout(pad=3, rect=(.05, 0, .95, .95))

		cmap, norm = PRF_colorbar_ax(cax)

		mean = hmaps.mean.data
		var = hmaps.var.data
		ff = hmaps.fanofactor.data
		mean_pw = hmaps.mean_pre_w.data
		var_pw = hmaps.var_pre_w.data
		ff_pw = hmaps.fanofactor_pre_w.data

		dom = hmaps.mean.bins

		kwargs = {'dom': dom, 'cmap': cmap, 'norm': norm, 'annot': annot_hm}

		heatmap_ax(ax1, mean, **kwargs)
		heatmap_ax(ax2, var, **kwargs)
		heatmap_ax(ax3, ff, **kwargs)
		heatmap_ax(ax4, mean_pw, **kwargs)
		heatmap_ax(ax5, var_pw, **kwargs)
		heatmap_ax(ax6, ff_pw, **kwargs)


		ax1.set_title('mean',		    fontsize=12, y=1.05)
		ax2.set_title('variance',       fontsize=12, y=1.05)
		ax3.set_title('fano factor', 	fontsize=12, y=1.05)

		# abuse y axis label for row title
		ax1.set_ylabel('weighted',		fontsize=12, labelpad=10)
		ax4.set_ylabel('unweighted',	fontsize=12, labelpad=10)

		return fig

	hmaps_vv = {
		0: [[hmaps]],
		1: [[hm_] for hm_ in hmaps],
		2: hmaps
	}[hmaps.ndim]

	for i, hmaps_v in enumerate(hmaps_vv):
		for j, hmaps_ in enumerate(hmaps_v):
			base_name = in_fname
			if is_filt_param(vary_param_1):
				base_name = '{}__{}_{}'.format(
					base_name,
					vary_param_1[0], vary_param_1[1][i]
				)
			elif is_weight_func(vary_param_1):
				base_name = '{}__{}_{}'.format(
					base_name,
					legend_labels_1[0], legend_labels_1[1][i]
				)

			if is_filt_param(vary_param_2):
				base_name = '__{}{}_{}'.format(
					base_name,
					vary_param_2[0], vary_param_2[1][j]
				)
			elif is_weight_func(vary_param_2):
				base_name = '__{}{}_{}'.format(
					base_name,
					'weight_func', legend_labels_2[j]
				)
			fig = make_hmap_fig(hmaps_)
			fig.suptitle(in_fname)
			fig.savefig('{}/{}.png'.format(out_dir, base_name))
			plt.close(fig)


def variance_fig(
		data,
		filt_params,
		vary_param_1,
		vary_param_2,
        out_filename,
		legend_labels_1,
		legend_labels_2,
		filename
):
	print 'plotting stats curves...'
	fig = plt.figure(figsize=(14, 8), tight_layout=True)

	label_kwargs = {
		'rotation': 0,
		'ha': 'right',
		'va': 'center',
		'labelpad': 10,
	}

	fname_ax =  plt.subplot2grid((5, 9), (0, 0), rowspan=1, colspan=3)
	params_ax = plt.subplot2grid((5, 9), (1, 0), rowspan=3, colspan=3)

	ax1 = plt.subplot2grid((5, 9), (0, 3), colspan=6)
	ax2 = plt.subplot2grid((5, 9), (1, 3), colspan=6, sharex=ax1)
	ax3 = plt.subplot2grid((5, 9), (2, 3), colspan=6, sharex=ax1, sharey=ax2)
	ax4 = plt.subplot2grid((5, 9), (3, 3), colspan=6, sharex=ax1)
	ax5 = plt.subplot2grid((5, 9), (4, 3), colspan=6, sharex=ax1, sharey=ax4)

	filenames_table(fname_ax, [filename, out_filename])
	filt_params_table(params_ax, filt_params)

	ax1.set_ylabel('norm of mean', **label_kwargs)
	ax2.set_ylabel('local variance', **label_kwargs)
	ax3.set_ylabel('global variance', **label_kwargs)
	ax4.set_ylabel('local fano factor', **label_kwargs)
	ax5.set_ylabel('global fano factor', **label_kwargs)

	if legend_labels_1:
		title, ticks = legend_labels_1
		ax5.set_xlabel(title)
		ax5.set_xticklabels(ticks)
		ax5.set_xticks(np.arange(len(vary_param_1[1])))
	else:
		ax5.set_xlabel(vary_param_1[0])


	def plot_stats_curves(norm_data):

		mean = [d.mean for d in norm_data]
		gvar = [d.gvar for d in norm_data]
		gff = [d.gfanofactor for d in norm_data]
		lvar = [d.lvar for d in norm_data]
		lff = [d.lfanofactor for d in norm_data]

		if not is_weight_func(vary_param_1):
			x = vary_param_1[1]
		else:
			x = np.arange(len(vary_param_1[1]))

		l, = ax1.plot(x, mean, '--o')
		ax2.plot(x, lvar, '--o')
		ax3.plot(x, gvar, '--o')
		ax4.plot(x, lff, '--o')
		ax5.plot(x, gff, '--o')

		return l		# for legend



	if vary_param_2:

		if legend_labels_2:
			label_list = legend_labels_2
		else:
			label_list = [
				'{} = {}'.format(vary_param_2[0], str(val))
				for val in vary_param_2[1]
			]

		line_list = []
		for i, norm_data in enumerate(data.T):
			l = plot_stats_curves(norm_data)
			line_list.append(l)

		fig.legend(
			line_list, label_list,
			'lower left',
			borderaxespad=3, borderpad=1
		)

	else:
		plot_stats_curves(data)

	for ax in [ax1, ax2, ax3, ax4, ax5]:
		ax.grid()
		ax.set_ylim(bottom=0)
		if ax is not ax5:
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_xticklines(), visible=False)

	fig.savefig(out_filename)