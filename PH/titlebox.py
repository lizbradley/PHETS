import numpy as np

def filename_table(ax, filename):
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')
	title_table = ax.table(
		cellText=[
			[filename.split('/')[-1]],   # remove leading "datasets/"
		],
		bbox=[0, 0, 1, 1],
		cellLoc='center'
	)
	# title_table.auto_set_font_size(False)
	# title_table.auto_set_font_size(8)

def filenames_table(ax, filenames):
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')
	title_table = ax.table(
		cellText=[
			[filenames[0].split('/')[-1]],   # remove leading "datasets/"
			[filenames[1].split('/')[-1]]   # remove leading "datasets/"
		],
		bbox=[0, 0, 1, 1],
		cellLoc='center'
	)
	# title_table.auto_set_font_size(False)
	# title_table.auto_set_font_size(8)


def filt_params_table(subplot, filt_params, fontsize=None):
	subplot.axis('off')
	subplot.set_xlim([0,1])
	subplot.set_ylim([0,1])


	display_params = (
		"max_filtration_param",
		"min_filtration_param",
		'num_divisions',
		"start",
		"worm_length",
		"ds_rate",
		"landmark_selector",
		"d_orientation_amplify",
		"d_use_hamiltonian",
		"d_cov",
		"simplex_cutoff",
		"use_cliques",
		"use_twr",
		"m2_d",
		"straight_VB",
		"dimension_cutoff",
		"graph_induced"
	)
	param_data = np.array([[key, filt_params[key]] for key in display_params])
	param_table = subplot.table(
		cellText=param_data,
		colWidths=[1.5, .5],
		bbox=[0, 0, 1, 1],  # x0, y0, width, height
	)

	param_table.auto_set_font_size(False)
	param_table.set_fontsize(6)



def update_epsilon(ax,e):
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')
	time_table = ax.table(
		cellText= [['$\epsilon$', '{:.6f}'.format(e)]],
		bbox=[0, 0, 1, 1],    # x0, y0, width, height
		colWidths=[.5, 1],
		cellLoc='center',

		animated=True,
	)
	# time_table.auto_set_font_size(False)
	# time_table.set_fontsize(8)



def movie_params_table(ax, params):
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')

	table = ax.table(
		cellText=[
			['color scheme',params[0]],
			['alpha', params[1]],
			# ['camera angle',params[2]],
		],
		bbox=[0, 0, 1, 1],    # x0, y0, width, height
		colWidths=[1.5, .5]
	)
	table.auto_set_font_size(False)
	table.set_fontsize(6)
