def title_table_window(ax, name, window):
	ax.axis('off')
	ax.axis('off')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])

	name_table = ax.table(
		cellText=[[name]],
		bbox=[0, .5, 1, .5],
		cellLoc='center'
	)
	window_table = ax.table(
		cellText=[['window #', window]],
		bbox=[0, 0, 1, .5],
		cellLoc='center'
	)


def title_table(ax, name):
	ax.axis('off')
	ax.axis('off')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])

	name_table = ax.table(
		cellText=[[name]],
		bbox=[0, 0, 1, 1],
		cellLoc='center',
		loc='center'
	)


def param_table(ax, params):

	ax.axis('off')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])

	try:
		if isinstance(params['tau'], float):
			params['tau'] = '{:6f}'.format(params['tau'])
	except:
		pass

	params_arr = [[key, item] for key, item in params.iteritems()]

	table = ax.table(
		cellText=params_arr,
		colWidths=[1, 1],
		bbox=[0, 0, 1, 1],  # x0, y0, width, height
	)


def slide_window_title(fname_ax, param_ax, traj, window):
	""" for slide window movies """

	title_table_window(fname_ax, traj.name, window)

	params = {
		'tau': traj.embed_params['tau'],
		'm': traj.embed_params['m'],
		'crop': traj.crop_lim,
		'time_units': traj.time_units,
		'window_length': traj.window_length,
		'window_step': traj.num_windows,
	}

	param_table(param_ax, params)


def vary_tau_title(fname_ax, param_ax, traj):
	""" for slide window movies """

	title_table(fname_ax, traj.name)

	params = {
		'tau': traj.embed_params['tau'],
		'm': traj.embed_params['m'],
		'time_units': traj.time_units,
	}

	param_table(param_ax, params)

def compare_vary_tau_title(ax, traj1, traj2, tau):

	ax.axis('off')
	ax.axis('off')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])

	ax.table(
		cellText=[
			['L', traj1.name],
		    ['R', traj2.name]
		],
		bbox=[0, .6, 1, .2],
		cellLoc='center',
		loc='center',
		colWidths=[1, 3],
	)

	ax.table(
		cellText=[['tau', tau]],
		bbox=[0, .4, 1, .1],
		cellLoc='center',
		loc='center',
		colWidths=[1, 3],
	)
