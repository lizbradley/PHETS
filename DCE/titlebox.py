
def title_table(ax, name, window):
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
	# title_table.auto_set_font_size(False)
	# title_table.auto_set_font_size(8)

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
	# param_table.auto_set_font_size(False)
	# param_table.set_fontsize(6)


