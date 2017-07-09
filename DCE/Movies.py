import DCE
from MovieTools import frames_to_movie
import Plotter
import numpy as np
import sys

from Utilities import pwd

from Tools import auto_tau

from MovieTools import remove_old_frames
from MovieTools import prep_save_worms_single, save_worms_single, prep_save_worms_double, save_worms_double

WAV_SAMPLE_RATE = 44100

def slide_window(
		in_filename,
		out_filename,
		window_size=.5,    # seconds
		step_size=.1,      # seconds
		tau=10,
		ds_rate=1,
		max_frames=0,      # 0 for disabled
		save_worms=True,
		save_movie=True
	):

	if save_worms: prep_save_worms_single()


	remove_old_frames()

	worm_length = sum(1 for line in open(in_filename)) / WAV_SAMPLE_RATE
	num_frames = worm_length/step_size

	for i, start in enumerate(np.arange(0, worm_length, step_size)):
		print 'frame %i of %i' % (i, num_frames)

		embed_crop = [start, start + window_size]
		sig = np.loadtxt(in_filename)
		DCE.embed(sig, 'DCE/temp_data/embedded_coords.txt', embed_crop, tau, 2, ds_rate=ds_rate)

		if save_worms: save_worms_single('{:d}-{}'.format(i, in_filename), i, tau, embed_crop)

		if save_movie:
			Plotter.make_window_frame('DCE/temp_data/embedded_coords.txt', in_filename, 'DCE/frames/frame%03d.png' % i, embed_crop, tau, i)

	if save_movie:
		frames_to_movie(out_filename, framerate=1)

def vary_tau(
		in_filename,
		out_filename,
		tau_lims=(1, 15),
		tau_inc=1,
		embed_crop=(1, 2),
		ds_rate=1,
		save_worms=True,
		save_movie=True,
		m=2,
	):

	remove_old_frames()

	if save_worms: prep_save_worms_single()

	for i, tau in enumerate(np.arange(tau_lims[0], tau_lims[1], tau_inc)):
		print 'frame %i of %i' % (i + 1, int((tau_lims[1] - tau_lims[0]) / tau_inc))
		sig = np.loadtxt(in_filename)
		DCE.embed(sig, 'DCE/temp_data/embedded_coords.txt', embed_crop, tau, m,  ds_rate=ds_rate)

		if save_worms: save_worms_single('{:d}-{}'.format(i, in_filename), i, int(tau), embed_crop)

		if save_movie: Plotter.make_window_frame('DCE/temp_data/embedded_coords.txt', in_filename, 'DCE/frames/frame%03d.png' % i, embed_crop, tau, i)

	if save_movie:
		frames_to_movie(out_filename, framerate=1)


def compare_vary_tau(
		in_filename_1,
		in_filename_2,
		out_filename,
		tau_lims,
		tau_inc=1,
		embed_crop=(1, 2),
		ds_rate=1,
		m=2,
		save_worms=True,
		save_movie=True
	):

	remove_old_frames()

	if save_worms: prep_save_worms_double()

	for i, tau in enumerate(np.arange(tau_lims[0], tau_lims[1], tau_inc)):
		print 'frame %i of %i' % (i + 1, int((tau_lims[1] - tau_lims[0]) / tau_inc))
		sig_1, sig_2 = np.loadtxt(in_filename_1), np.loadtxt(in_filename_2)
		DCE.embed(sig_1, 'DCE/temp_data/embedded_coords_comp1.txt', embed_crop, tau, m, ds_rate=ds_rate)
		DCE.embed(sig_2, 'DCE/temp_data/embedded_coords_comp2.txt', embed_crop, tau, m, ds_rate=ds_rate)

		if save_worms: save_worms_double('{:d}-txt_wave_file1'.format(i), '{:d}-txt_wave_file2'.format(i), i, tau, tau, embed_crop, embed_crop)

		if save_movie:
			Plotter.compare_vary_tau_frame('DCE/frames/frame%03d.png' % i, in_filename_1, in_filename_2, i, tau, embed_crop)

	if save_movie:
		frames_to_movie(out_filename, framerate=1)


def get_params_table(args_dict):
	params_table = [
		['i_lims', args_dict['i_lims']],
		['embed_crop_1', args_dict['embed_crop_1']],
		['embed_crop_2', args_dict['embed_crop_2']],
		['auto_crop_length', args_dict['auto_crop_length']],
		['tau_1', args_dict['tau_1']],
		['tau_2', args_dict['tau_2']],
		['tau_T', '{:.5f}'.format(args_dict['tau_T'])],
		['normalize_volume', args_dict['normalize_volume']],
		['save_worms', args_dict['save_worms']],
		['save_movie', args_dict['save_movie']],
	]

	return params_table



def get_comp_tables(*args):
	ideal_table = [['f ideal (Hz)', args[0]]]

	title_1 = [[args[1].split('/')[-1]]]
	title_2 = [[args[2].split('/')[-1]]]


	f_detect_1 = '{:.5f}'.format(args[3]) if isinstance(args[3], float) else args[3]
	f_detect_2 = '{:.5f}'.format(args[4]) if isinstance(args[4], float) else args[4]



	table_1 = [['f detect (Hz)', f_detect_1],
			   ['tau (sec)', '{:.5f}'.format(args[5])],
			   ['tau (samp)', int(args[5] * WAV_SAMPLE_RATE)],
			   ['crop (sec)', '({:.3f}, {:.3f})'.format(args[7][0], args[7][1])]]

	table_2 = [['f detect (Hz)', f_detect_2],
			   ['tau (sec)', '{:.5f}'.format(args[6])],
			   ['tau (samp)', int(args[6] * WAV_SAMPLE_RATE)],
			   ['crop (sec)', '({:.3f}, {:.3f})'.format(args[8][0], args[8][1])]]

	return [ideal_table, title_1, title_2, table_1, table_2]





from Tools import auto_crop


def compare_multi(
		dir1, dir1_base,
		dir2, dir2_base,
		out_filename,

		i_lims=(1, 89),

		embed_crop_1='auto',    # (start, stop) in seconds or 'auto'
		embed_crop_2='auto',
		auto_crop_length=.3,		# for when embed_crop = 'auto'

		tau_1='auto ideal',       # seconds or 'auto detect' or 'auto ideal'
		tau_2='auto ideal',
		tau_T=1/np.pi,          # tau_sec = period * tau_T

		save_worms=True,
		save_movie=True,

		normalize_volume=True,

		waveform_zoom=None,
		ds_rate=1, dpi=200, m=2
	):
	"""makes frames for comparison movie: proportional tau, constant, vary in files"""

	params_table = get_params_table(locals())

	tau_1_cmd, tau_2_cmd = tau_1, tau_2

	if save_worms: prep_save_worms_double()

	remove_old_frames()
	frame_idx = 0





	for i in xrange(i_lims[0], i_lims[1]):

		frame_idx += 1
		print
		print 'frame', frame_idx

		# load files #
		filename_1 = dir1 + "/%02d" % i + dir1_base
		filename_2 = dir2 + "/%02d" % i + dir2_base
		sig_1 = np.loadtxt(filename_1)
		sig_2 = np.loadtxt(filename_2)


		if normalize_volume: sig_1 = sig_1 / np.max(np.abs(sig_1))
		if normalize_volume: sig_2 = sig_2 / np.max(np.abs(sig_2))

		crop_1 = auto_crop(embed_crop_1, sig_1, auto_crop_length)
		crop_2 = auto_crop(embed_crop_2, sig_2, auto_crop_length)

		f_ideal, f_disp_1, tau_1 = auto_tau(tau_1_cmd, sig_1, i, tau_T, crop_1, filename_1)
		f_ideal, f_disp_2, tau_2 = auto_tau(tau_2_cmd, sig_2, i, tau_T, crop_2, filename_2)

		computed_tables = get_comp_tables(f_ideal, filename_1, filename_2, f_disp_1, f_disp_2, tau_1, tau_2, crop_1, crop_2)

		DCE.embed(sig_1, 'DCE/temp_data/embedded_coords_comp1.txt', crop_1, tau_1, m, ds_rate=ds_rate)
		DCE.embed(sig_2, 'DCE/temp_data/embedded_coords_comp2.txt', crop_2, tau_2, m, ds_rate=ds_rate)

		if save_worms: save_worms_double(filename_1, filename_2, i, tau_1, tau_2, crop_1, crop_2)

		title_tables = [params_table, computed_tables]
		if save_movie:
			Plotter.compare_multi_frame(frame_idx, sig_1, sig_2, filename_1, filename_2, crop_1, crop_2, dpi, title_tables)

	if save_movie:
		frames_to_movie(out_filename, framerate=1)



