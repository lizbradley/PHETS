import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PH.plots import heatmap_ax
from PH.titlebox import filenames_table, filt_params_table
from utilities import clear_dir


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
		data_arr,
		data_arr_pre_weight,
		filt_params,
		vary_param_1,
        vary_param_2,
		legend_labels,
		filename,
		annot_hm,
		unit_test=False
):

	if unit_test:
		out_dir = 'output/heatmaps/'
		return
	else:
		out_dir = 'output/PRFstats/heatmaps/'

	if not clear_dir(out_dir):
		print 'skipping heatmaps'
		return

	print 'plotting heatmaps...'

	def make_hmap_fig(hmap_data, hmap_data_pw):

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


		x = y = np.linspace(0, np.power(2, .5), filt_params['num_divisions'])

		heatmap_ax(ax1, cax, x, y, hmap_data.pointwise_mean, annot_hm)
		heatmap_ax(ax2, cax, x, y, hmap_data.pointwise_var, annot_hm)
		heatmap_ax(ax3, cax, x, y, hmap_data.functional_COV, annot_hm)
		heatmap_ax(ax4, cax, x, y, hmap_data_pw.pointwise_mean, annot_hm)
		heatmap_ax(ax5, cax, x, y, hmap_data_pw.pointwise_var, annot_hm)
		heatmap_ax(ax6, cax, x, y, hmap_data_pw.functional_COV, annot_hm)


		ax1.set_title('pointwise mean',		fontsize=12, y=1.05)
		ax2.set_title('pointwise variance', fontsize=12, y=1.05)
		ax3.set_title('functional COV', 	fontsize=12, y=1.05)

		# abuse y axis label for row title
		ax1.set_ylabel('weighted',		fontsize=12, labelpad=10)
		ax4.set_ylabel('unweighted',	fontsize=12, labelpad=10)

		ticks = np.linspace(0, 1.4, filt_params['num_divisions'], endpoint=True)
		while len(ticks) > 6:
			ticks = ticks[1::2]
		for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
			ax.xaxis.set_ticks(ticks)
			ax.yaxis.set_ticks(ticks)

		return fig


	if vary_param_2:
		for i, val_2 in enumerate(vary_param_2[1]):
			for j, val_1 in enumerate(vary_param_1[1]):
				data = data_arr[i, j]
				if vary_param_2[0] == 'weight_func':
					data_pw = data_arr_pre_weight[0, j]
				else:
					data_pw = data_arr_pre_weight[i, j]
				fig = make_hmap_fig(data, data_pw)
				fig.suptitle(filename.split('/')[-1])
				if legend_labels:
					val_2 = legend_labels[i]
				fname = '{}_{}__{}_{}.png'.format(
					vary_param_2[0], val_2, vary_param_1[0], val_1
				)
				fig.savefig(out_dir + fname)
				plt.close(fig)

	else:
		data_arr = data_arr[0]
		data_arr_pre_weight = data_arr_pre_weight[0]
		for j, val_1 in enumerate(vary_param_1[1]):
			data = data_arr[j]
			data_pw = data_arr_pre_weight[j]
			fig = make_hmap_fig(data, data_pw)
			fig.suptitle(filename.split('/')[-1])
			fname = '{}_{}.png'.format(vary_param_1[0], val_1)
			fig.savefig(out_dir + fname)
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
	print 'plotting variance curves...'
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

		if not callable(vary_param_1[1][0]):
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