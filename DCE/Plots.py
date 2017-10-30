import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import TitleBox
from config import WAV_SAMPLE_RATE


def plot_dce(fig, ax, dce_data):
	# print 'plotting dce...'

	# ax.set_aspect('equal', adjustable='box', anchor='C')
	# ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	# ax.use_sticky_edges = False



	amb_dim = dce_data.shape[1]

	if amb_dim == 2:

		x = dce_data[:, 0]
		y = dce_data[:, 1]
		ax.scatter(x, y, color='black', s=.1)

	elif amb_dim == 3:
		x = dce_data[:, 0]
		y = dce_data[:, 1]
		z = dce_data[:, 2]

		ax.scatter(x, y, z, color='black', s=.1)
		# ax.set_axis('off')


	ax.set(aspect='equal', adjustable='datalim', anchor='C')

	# downsample auto ticks
	# yticks = ax.get_yticks()
	# yticks = yticks[::2]
	# xticks = ax.get_xticks()
	# xticks = xticks[::2]
	# ax.set_yticks(yticks)
	# ax.set_xticks(xticks)

	# choose second outermost auto ticks
	yticks = ax.get_yticks()
	ymin, ymax = yticks[1], yticks[-2]
	xticks = ax.get_xticks()
	xmin, xmax = xticks[1], xticks[-2]
	ax.set_yticks([ymin, ymax])
	ax.set_xticks([xmin, xmax])


	fig.subplots_adjust(left=.07, bottom=.07, right=.93, top=.93, wspace=.5, hspace=.5)





def plot_signal_zoom(ax, full_sig, crop, time_units='seconds', sig=None):

	if full_sig is None:
		x = np.linspace(crop[0], crop[1], len(sig))
		y = sig

	else:
		# backwards compatibility #
		if time_units == 'samples':
			x = np.arange(crop[0], crop[1])
		elif time_units == 'seconds':
			crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
			x = np.linspace(0, len(full_sig) / WAV_SAMPLE_RATE, len(full_sig))[crop[0]:crop[1]]
		else:
			print 'ERROR: invalid time_units'
			sys.exit()

		y = full_sig[crop[0]:crop[1]]


	ax.plot(x, y, color='k', zorder=0, lw= .5)


def plot_title(fname_ax, param_ax, title_info):
	""" for slide window movies """

	TitleBox.add_fname_table(fname_ax, title_info)
	TitleBox.add_param_table(param_ax, title_info)



def make_frame(traj, sig, window, frame_fname, title_info, time_units=None):
	fig = plt.figure(figsize=(10, 8), tight_layout=False, dpi=100)
	# fig.subplots_adjust(hspace=.5)
	m = title_info['m']

	fname_ax = 				plt.subplot2grid((8, 10), (0, 0), colspan=3)
	param_ax =				plt.subplot2grid((8, 10), (1, 0), colspan=3, rowspan=3)
	sig_ax = 				plt.subplot2grid((8, 10), (6, 0), colspan=10, rowspan=2)
	if m == 2: dce_ax =		plt.subplot2grid((8, 10), (0, 4), colspan=6, rowspan=6)
	else: dce_ax = 			plt.subplot2grid((8, 10), (0, 4), colspan=6, rowspan=6, projection='3d')

	# if m == 2: dce_ax =		fig.add_axes([.4, .3, .6, .6])
	# else: dce_ax =			fig.add_axes([.4, .3, .6, .6], projection='3d')

	if title_info['crop']:
		t_offset = title_info['crop'][0]
	else:
		t_offset = 0


	plot_title(fname_ax, param_ax, title_info)
	plot_dce(fig, dce_ax, traj)
	plot_signal(sig_ax, sig, window, offset=t_offset, time_units=time_units)
	plt.savefig(frame_fname)
	plt.close(fig)



def compare_vary_tau_frame(out_file_name, wave_file_name1, wave_file_name2, frame_num, tau, embed_crop, m):
	fig = plt.figure(figsize=(12, 9), tight_layout=True)
	if m == 2:
		subplot1 = plt.subplot2grid((5, 2), (0, 0), rowspan=4)
		subplot2 = plt.subplot2grid((5, 2), (0, 1), rowspan=4)
	elif m == 3:
		subplot1 = plt.subplot2grid((5, 2), (0, 0), rowspan=4, projection='3d')
		subplot2 = plt.subplot2grid((5, 2), (0, 1), rowspan=4, projection='3d')
	else:
		print 'ERROR: m must be 2 or 3'
		sys.exit()

	subplot3 = plt.subplot2grid((5, 2), (4,0))
	subplot4 = plt.subplot2grid((5, 2), (4, 1), sharey=subplot3)
	plt.setp(subplot4.get_yticklabels(), visible=False)

	plot_dce(subplot1, 'DCE/temp/embedded_coords_comp1.txt')
	plot_dce(subplot2, 'DCE/temp/embedded_coords_comp2.txt')

	wave_data1, wave_data2 = np.loadtxt(wave_file_name1), np.loadtxt(wave_file_name2)
	plot_signal(subplot3, wave_data1, embed_crop)
	plot_signal(subplot4, wave_data2, embed_crop)

	subplot1.set_title(wave_file_name1)
	subplot2.set_title(wave_file_name2)
	fig.suptitle('$tau = %d$' % tau, bbox={'pad':5}, fontsize=14)

	plt.savefig(out_file_name)
	plt.close(fig)



def plot_titlebox(subplots, table_arr):
	[ax.axis('off') for ax in subplots]

	param_title, ideal_f_title, comp_title_1, comp_title_2 = subplots

	params_table = param_title.table(
		cellText=table_arr[0],
		# colWidths=col_widths,
		bbox=[0, 0, 1, 1],  # x0, y0, width, height
	)

	f_ideal, title_1_arr, title_2_arr, table_1_arr, table_2_arr = table_arr[1]

	ideal_table = ideal_f_title.table(
		cellText = f_ideal,
		bbox=[0, 0, 1, 1],
	)


	title_1 = comp_title_1.table(
		cellText=title_1_arr,
		cellLoc='center',
		bbox=[0, .8, 1, .2],  # x0, y0, width, height
	)
	table_1 = comp_title_1.table(
		cellText=table_1_arr,
		bbox=[0, 0, 1, .8],	# x0, y0, width, height
	)


	title_2 = comp_title_2.table(
		cellText=title_2_arr,
		cellLoc='center',
		bbox=[0, .8, 1, .2]	# x0, y0, width, height
	)
	table_2 = comp_title_2.table(
		cellText=table_2_arr,
		bbox=[0, 0, 1, .8]		# x0, y0, width, height
	)



def compare_multi_frame(frame_idx, sig1, sig2, crop_1, crop_2, dpi, title_tables, m):
	fig = plt.figure(figsize=(16, 9), tight_layout=True, dpi=dpi)


	param_title =		plt.subplot2grid((9, 16), (0, 0), rowspan=4, colspan=3)
	comp_title_1 =		plt.subplot2grid((9, 16), (4, 0), rowspan=2, colspan=3)
	comp_title_2 = 		plt.subplot2grid((9, 16), (6, 0), rowspan=2, colspan=3)
	ideal_f_title = 	plt.subplot2grid((9, 16), (8, 0), rowspan=1, colspan=3)

	if m == 2:
		ax1 = 			plt.subplot2grid((9, 16), (0, 4), rowspan=5, colspan=5)								# dce 1
		ax2 = 			plt.subplot2grid((9, 16), (0, 10), rowspan=5, colspan=5)								# dce 2
	elif m == 3:
		ax1 = 			plt.subplot2grid((9, 16), (0, 4), rowspan=5, colspan=5, projection='3d')				# dce 1
		ax2 = 			plt.subplot2grid((9, 16), (0, 10), rowspan=5, colspan=5, projection='3d')			# dce 2
	else:
		print 'ERROR: m must be 2 or 3'
		sys.exit()

	ax3 = 				plt.subplot2grid((9, 16), (5, 4), colspan=5, rowspan=2)					# waveform full 1
	ax4 = 				plt.subplot2grid((9, 16), (5, 10), colspan=5, rowspan=2, sharey=ax3)		# waveform full 2

	ax5 = 				plt.subplot2grid((9, 16), (7, 4), colspan=5, rowspan=2)					# waveform zoom 1
	ax6 = 				plt.subplot2grid((9, 16), (7, 10), colspan=5, rowspan=2, sharey=ax5)		# waveform zoom 2


	title_plots = [param_title, ideal_f_title, comp_title_1, comp_title_2]

	plot_titlebox(title_plots, title_tables)

	plot_dce(ax1, 'DCE/temp/embedded_coords_comp1.txt')
	plot_dce(ax2, 'DCE/temp/embedded_coords_comp2.txt')

	plot_signal(ax3, sig1, crop_1)
	plot_signal(ax4, sig2, crop_2)

	plot_signal_zoom(ax5, sig1, crop_1)
	plot_signal_zoom(ax6, sig2, crop_2)

	out_filename = 'DCE/frames/frame%03d.png' % frame_idx
	plt.savefig(out_filename)
	plt.close(fig)


def plot_signal(out, waveform_data, window=None, time_units='seconds', offset=None):

	if isinstance(out, basestring):
		fig = plt.figure(figsize=(4, 1))
		ax = fig.add_subplot(111)
	else:
		ax = out

	y = waveform_data

	if time_units == 'samples':
		x = np.arange(0, len(y))
		ax.set_xlabel('time (samples)')

	else: 			# seconds
		x = np.linspace(0, len(y) / WAV_SAMPLE_RATE, len(y))
		ax.set_xlabel('time (seconds)')

	if offset:
		x = x + offset
		window = np.array(window) + offset

	ax.plot(x, y, color='k', zorder=0, lw= .5)
	ax.axis('tight')

	if window is not None:
		if math.fabs(window[0] - window[1]) < .01:
			ax.axvline(window[0], color='r', alpha=0.7, zorder=1)
		ax.axvspan(window[0], window[1], facecolor='r', alpha=0.5, zorder=1)

	ymin, ymax = ax.get_ylim()
	ylim = abs(ymin) if abs(ymin) >= abs(ymax) else abs(ymax)
	ax.set_ylim([-ylim, ylim])
	ax.set_yticks([-ylim, 0, ylim])
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))


	if isinstance(out, basestring): plt.savefig(out)