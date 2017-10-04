
def add_fname_table(ax, title_info):
	ax.axis('off')

	if title_info['title']:
		title_table = ax.table(
			cellText=[
				[title_info['title'].split('/')[-1]],   # remove leading "datasets/"
			],
			bbox=[0, 0, 1, 1],
			cellLoc='center'
		)

	elif isinstance(title_info['fname'], basestring):
		title_table = ax.table(
			cellText=[
				[title_info['fname'].split('/')[-1]],   # remove leading "datasets/"
			],
			bbox=[0, 0, 1, 1],
			cellLoc='center'
		)
		# title_table.auto_set_font_size(False)
		# title_table.auto_set_font_size(8)

def add_param_table(ax, params):

	ax.axis('off')
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])

	try:
		if isinstance(params['tau'], float):
			params['tau'] = '{:6f}'.format(params['tau'])
	except:
		pass



	param_data = [[key, params[key]] for key in params if key not in ('fname', 'title')]
	param_table = ax.table(
		cellText=param_data,
		colWidths=[1, 1],
		bbox=[0, 0, 1, 1],  # x0, y0, width, height
	)
	# param_table.auto_set_font_size(False)
	# param_table.set_fontsize(6)


