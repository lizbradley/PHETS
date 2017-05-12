import math
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import FormatStrFormatter

WAV_SAMPLE_RATE = 44100.

# noinspection PyTypeChecker
def plot_dce(ax, in_file_name):
	# print 'plotting dce...'
	dce_data = np.loadtxt(in_file_name)
	x = dce_data[:,0]
	y = dce_data[:,1]
	ax.scatter(x, y, color='black', s=.1)
	# subplot.set_aspect('equal')
	ax.set(adjustable='box-forced', aspect='equal')

	# x0,x1 = ax.get_xlim()
	# y0,y1 = ax.get_ylim()
	# ax.set_aspect(abs(x1-x0)/abs(y1-y0))

	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	# xlims = ax.get_xlim()
	# ax.set_xticks([0, xlims[1]])
	#
	# ylims = ax.get_ylim()
	# ax.set_yticks([0, ylims[1]])



def plot_waveform(subplot, waveform_data, crop):
	y = waveform_data
	x = np.linspace(0, len(y) / WAV_SAMPLE_RATE, len(y))

	subplot.plot(x, y, color='k', zorder=0, lw= .5)
	subplot.axis('tight')
	if math.fabs(crop[0] - crop[1]) < .01:   # how to un-hardcode?
		subplot.axvline(crop[0], color='r', alpha=0.7, zorder=1)
	subplot.axvspan(crop[0], crop[1], facecolor='r', alpha=0.5, zorder=1)

	subplot.set_xlabel('time (s)')



def plot_waveform_zoom(ax, full_sig, crop):
	crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
	y = full_sig[crop[0]:crop[1]]
	x = np.linspace(0, len(full_sig) / WAV_SAMPLE_RATE, len(full_sig))[crop[0]:crop[1]]

	# x0,x1 = ax.get_xlim()
	# y0,y1 = ax.get_ylim()
	# ax.set_aspect(abs(x1-x0)/abs(y1-y0))

	ax.plot(x, y, color='k', zorder=0, lw= .5)

	ax.axis('tight')

	ax.set_xlabel('time (s)')

	# subplot.set_ylim(-1.1, 1.1)
	# subplot.set_yticks([-1, 0, 1])

def plot_waveform_zoom_only(ax, full_sig, crop, fname):

	crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
	y = full_sig[crop[0]:crop[1]]
	x = np.linspace(0, len(full_sig) / WAV_SAMPLE_RATE, len(full_sig))[crop[0]:crop[1]]

	# x0,x1 = ax.get_xlim()
	# y0,y1 = ax.get_ylim()
	# ax.set_aspect(abs(x1-x0)/abs(y1-y0))

	ax.plot(x, y, color='k', zorder=0, lw=.5)

	ax.axis('tight')

	ax.set_xlabel('time (s)')

# subplot.set_ylim(-1.1, 1.1)
# subplot.set_yticks([-1, 0, 1])



def plot_title(subplot, in_file_name, tau):
	""" for slide window movies """
	subplot.axis('off')
	subplot.set_xlim([0,1])
	subplot.set_ylim([0,1])

	tau_str = r'$\tau = %d$' % tau
	subplot.text(.5,.5 , tau_str,
				 horizontalalignment='center',
				 verticalalignment='center',
				 size=14,
				 bbox=dict(facecolor='none')
				 )




def make_window_frame(coords_file_name, wave_file_name, out_file_name, embed_crop, tau, frame_num):
	fig = pyplot.figure(figsize=(9, 9), tight_layout=False)
	fig.subplots_adjust(hspace=.5)
	fig.suptitle(wave_file_name)
	title_subplot = pyplot.subplot2grid((4, 4), (0, 3), rowspan=3)
	dce_subplot = pyplot.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
	wavform_subplot = pyplot.subplot2grid((4, 4), (3, 0), colspan=4)

	wave_data = np.loadtxt(wave_file_name)

	plot_dce(dce_subplot, coords_file_name)
	plot_waveform(wavform_subplot, wave_data, embed_crop)
	plot_title(title_subplot, wave_file_name, tau)
	pyplot.savefig(out_file_name)
	pyplot.close(fig)


def compare_vary_tau_frame(out_file_name, wave_file_name1, wave_file_name2, frame_num, tau, embed_crop):
	fig = pyplot.figure(figsize=(12, 9), tight_layout=True)
	subplot1 = pyplot.subplot2grid((5, 2), (0,0), rowspan=4)
	subplot2 = pyplot.subplot2grid((5, 2), (0,1), rowspan=4)
	subplot3 = pyplot.subplot2grid((5, 2), (4,0))
	subplot4 = pyplot.subplot2grid((5, 2), (4, 1), sharey=subplot3)
	pyplot.setp(subplot4.get_yticklabels(), visible=False)

	plot_dce(subplot1, 'DCE/temp_data/embedded_coords_comp1.txt')
	plot_dce(subplot2, 'DCE/temp_data/embedded_coords_comp2.txt')

	wave_data1, wave_data2 = np.loadtxt(wave_file_name1), np.loadtxt(wave_file_name2)
	plot_waveform(subplot3, wave_data1, embed_crop)
	plot_waveform(subplot4, wave_data2, embed_crop)

	subplot1.set_title(wave_file_name1)
	subplot2.set_title(wave_file_name2)
	fig.suptitle('$tau = %d$' % tau, bbox={'pad':5}, fontsize=14)

	pyplot.savefig(out_file_name)
	pyplot.close(fig)



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



def compare_multi_frame(frame_idx, sig1, sig2, filename_1, filename_2, crop_1, crop_2, dpi, title_tables):
	fig = pyplot.figure(figsize=(16, 9), tight_layout=True, dpi=dpi)


	param_title =		pyplot.subplot2grid((9, 16), (0, 0), rowspan=4, colspan=3)
	comp_title_1 =		pyplot.subplot2grid((9, 16), (4, 0), rowspan=2, colspan=3)
	comp_title_2 = 		pyplot.subplot2grid((9, 16), (6, 0), rowspan=2, colspan=3)
	ideal_f_title = 	pyplot.subplot2grid((9, 16), (8, 0), rowspan=1, colspan=3)


	ax1 = 				pyplot.subplot2grid((9, 16), (0, 4), rowspan=5, colspan=5)					# dce 1
	ax2 = 				pyplot.subplot2grid((9, 16), (0, 10), rowspan=5, colspan=5)					# dce 2

	ax3 = 				pyplot.subplot2grid((9, 16), (5, 4), colspan=5, rowspan=2)					# waveform full 1
	ax4 = 				pyplot.subplot2grid((9, 16), (5, 10), colspan=5, rowspan=2, sharey=ax3)		# waveform full 2

	ax5 = 				pyplot.subplot2grid((9, 16), (7, 4), colspan=5, rowspan=2)					# waveform zoom 1
	ax6 = 				pyplot.subplot2grid((9, 16), (7, 10), colspan=5, rowspan=2, sharey=ax5)		# waveform zoom 2

	# ax1.set_position([0, 0, 1, 1])
	# ax2.set_position([0, 0, 1, 1])


	title_plots = [param_title, ideal_f_title, comp_title_1, comp_title_2]


	plot_titlebox(title_plots, title_tables)

	plot_dce(ax1, 'DCE/temp_data/embedded_coords_comp1.txt')
	plot_dce(ax2, 'DCE/temp_data/embedded_coords_comp2.txt')

	plot_waveform(ax3, sig1, crop_1)
	plot_waveform(ax4, sig2, crop_2)

	plot_waveform_zoom(ax5, sig1, crop_1)
	plot_waveform_zoom(ax6, sig2, crop_2)



	ax1.set_title(filename_1.split('/')[-1])
	ax2.set_title(filename_2.split('/')[-1])

	out_filename = 'DCE/frames/frame%03d.png' % frame_idx
	pyplot.savefig(out_filename)
	pyplot.close(fig)

