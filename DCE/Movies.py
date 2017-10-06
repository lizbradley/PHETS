import DCE
# from MovieTools import frames_to_movie
import Plots
import numpy as np
import sys

from Utilities import frames_to_movie

from Tools import auto_tau, auto_crop, crop_sig

from config import WAV_SAMPLE_RATE

from MovieTools import remove_old_frames
from MovieTools import prep_save_worms_double, save_worms_double



def get_data(fname):
	if isinstance(fname, basestring):
		print 'loading signal...'
		return np.loadtxt(fname)
	else:
		return fname	# is array


def slide_window(
		in_filename,
		out_filename,
		time_units='samples',
		window_size=.5,    	# seconds
		window_step=.1,     # seconds
		tau=10,
		ds_rate=1,
		m=2,  				# embed dimension
		save_movie=True,
		crop=None,
		title=None,
		framerate=1
	):

	title_info = {
		'fname': in_filename,
		'tau': tau,
		'm': m,
		'crop': crop,
		'time_units': time_units,
		'window_size': window_size,
		'window_step': window_step,
		'title': title
	}

	if save_movie: remove_old_frames()

	full_sig = get_data(in_filename)

	sig = crop_sig(full_sig, crop, time_units)

	if time_units == 'seconds':
		worm_length = len(sig) / WAV_SAMPLE_RATE
	else:
		worm_length = len(sig)

	frame_fname = 'DCE/frames/frame%03d.png'
	trajs = []
	print 'building movie...'
	for i, start in enumerate(np.arange(0, worm_length, window_step)):
		print 'frame %i of %i' % (i, worm_length / window_step)
		window = [start, start + window_size]
		traj = DCE.embed(
			sig, tau, m, crop=window, ds_rate=ds_rate, time_units=time_units
		)
		trajs.append(traj)
		title_info.update({'frame #': i})

		if save_movie:
			Plots.make_frame(traj, sig, window, frame_fname % i, title_info, time_units=time_units)

	if save_movie:
		frames_to_movie(out_filename, frame_fname,
						framerate=framerate, aspect=(10,8))

	return trajs




def vary_tau(
		in_filename,
		out_filename,
		tau_lims=(1, 15),
		tau_inc=1,
		crop=(1, 2),
		time_units='samples',
		ds_rate=1,
		m=2,
		save_movie=True,
		title=None,
		framerate=1
		
	):

		title_info = {
			'fname': in_filename,
			'tau_lims': tau_lims,
			'tau_inc': tau_inc,
			'm': m,
			'crop': crop,
			'time_units': time_units,
			'title': title,
		}

		if save_movie: remove_old_frames()
		full_sig = get_data(in_filename)
		sig = crop_sig(full_sig, crop, time_units)

		frame_fname = 'DCE/frames/frame%03d.png'
		trajs = []
		num_frames = int(np.floor((tau_lims[1] - tau_lims[0]) / tau_inc))
		print 'building movie...'
		for i, tau in enumerate(np.arange(tau_lims[0], tau_lims[1], tau_inc)):
			print 'frame {} of {}'.format(i, num_frames)
			traj = DCE.embed(sig, tau, m,
							 ds_rate=ds_rate, time_units=time_units)
			trajs.append(traj)
			title_info.update({'frame #': i})
			title_info.update({'tau': tau})

			if save_movie:
				Plots.make_frame(traj, sig, crop, frame_fname % i, title_info)

		if save_movie:
			frames_to_movie(out_filename, frame_fname, framerate=framerate)

		return trajs

def compare_vary_tau(
		in_filename_1,
		in_filename_2,
		out_filename,
		tau_lims,
		tau_inc=1,
		embed_crop=(1, 2),
		ds_rate=1,
		m=2,
		save_trajectories=True,
		save_movie=True
	):

	remove_old_frames()

	if save_trajectories: prep_save_worms_double()

	for i, tau in enumerate(np.arange(tau_lims[0], tau_lims[1], tau_inc)):
		print 'frame %i of %i' % (i + 1, int((tau_lims[1] - tau_lims[0]) / tau_inc))
		sig_1, sig_2 = np.loadtxt(in_filename_1), np.loadtxt(in_filename_2)
		DCE.embed_v1(sig_1, 'DCE/temp/embedded_coords_comp1.txt', embed_crop, tau, m, ds_rate=ds_rate)
		DCE.embed_v1(sig_2, 'DCE/temp/embedded_coords_comp2.txt', embed_crop, tau, m, ds_rate=ds_rate)

		if save_trajectories: save_worms_double('{:d}-txt_wave_file1'.format(i), '{:d}-txt_wave_file2'.format(i), i, tau, tau, embed_crop, embed_crop)

		if save_movie:
			Plots.compare_vary_tau_frame('DCE/frames/frame%03d.png' % i, in_filename_1, in_filename_2, i, tau, embed_crop, m)

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
		['save_trajectories', args_dict['save_trajectories']],
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

		save_trajectories=True,
		save_movie=True,

		normalize_volume=True,

		waveform_zoom=None,
		ds_rate=1, dpi=200, m=2
	):
	"""makes frames for comparison movie: proportional tau, constant, vary in files"""

	params_table = get_params_table(locals())

	tau_1_cmd, tau_2_cmd = tau_1, tau_2

	if save_trajectories: prep_save_worms_double()

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

		DCE.embed_v1(sig_1, 'DCE/temp/embedded_coords_comp1.txt', crop_1, tau_1, m, ds_rate=ds_rate)
		DCE.embed_v1(sig_2, 'DCE/temp/embedded_coords_comp2.txt', crop_2, tau_2, m, ds_rate=ds_rate)

		if save_trajectories: save_worms_double(filename_1, filename_2, i, tau_1, tau_2, crop_1, crop_2)

		title_tables = [params_table, computed_tables]
		if save_movie:
			Plots.compare_multi_frame(frame_idx, sig_1, sig_2, crop_1, crop_2, dpi, title_tables, m)

	if save_movie:
		frames_to_movie(out_filename, framerate=1)



