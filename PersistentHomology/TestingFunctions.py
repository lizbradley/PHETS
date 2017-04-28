import BuildComplex
import FiltrationPlotter
import PersistencePlotter
import numpy as np
import os
import sys
import time
from sys import platform

parameter_set = {
	"num_divisions": 50,
	"max_filtration_param": -20,
	"min_filtration_param": 0,
	"start": 0,
	"worm_length": 10000,
	"ds_rate": 50,
	"landmark_selector": "maxmin",
	"use_ne_for_maxmin": False,
	"d_speed_amplify": 1,
	"d_orientation_amplify": 1,
	"d_stretch": 1,
	"d_ray_distance_amplify": 1,
	"d_use_hamiltonian": 0,
	"d_cov" : 0,
	"simplex_cutoff": 0,
	"weak": False,
	"absolute": False,
	"use_cliques": False,
	"use_twr": False,
	"m2_d": 0,  #Set to anything but 0 to run, set 'time_order_landmarks' = TRUE (don't think i need last part anymore - CHECK)
	"straight_VB": 0, #
	"out": None,
	"program": "Perseus",
	"dimension_cutoff": 2,
	"time_order_landmarks": False,
	"connect_time_1_skeleton": False,
	"reentry_filter": False,
	"store_top_simplices": True,
	"sort_output": False,
	"graph_induced": False    # Use graph induced complex to build filtration.
}

def build_and_save_filtration(in_file_name, params, start=0):
    start_time = time.time()
    lines = open(in_file_name).readlines()
    start_idx = int(len(lines) * start)
    os.chdir('PersistentHomology')
    open('temp_data/worm_data.txt', 'w').writelines(lines[start_idx:])

    print "building filtration..."
    filtration = BuildComplex.build_filtration('temp_data/worm_data.txt', params)
    witness_coords = filtration[1][1]
    landmark_coords = filtration[1][0]
    complexes = sorted(list(filtration[0]))

    print "saving filtration..."
    np.save('temp_data/witness_coords.npy', witness_coords)
    np.save('temp_data/landmark_coords.npy', landmark_coords)
    np.save('temp_data/complexes.npy', complexes)
    os.chdir('..')
    print("build_and_save_filtration() time elapsed: %d seconds \n" % (time.time() - start_time))

def frames_to_movie(out_file_name, framerate):
    print 'building movie...'

    in_str = ('ffmpeg -y -framerate %i ' % framerate) + '-i frames/image%03d.png'
    out_str = (' -r %d ' % 24) + out_file_name

    os.system(in_str + out_str)
    print os.getcwd() + ('\\' if os.name == 'nt' else '/') + out_file_name


def check_saved_filtration(in_file_name):
    """checks that the first? last? line of saved array and text file match"""
    saved_witness_data = np.load('PersistentHomology/temp_data/witness_coords.npy')
    in_file_data = open(in_file_name)
    line_saved = saved_witness_data[1]
    line_in = in_file_data.readline().split()
    if len(line_saved) == len(line_in):
        same = True
        # for x_s, x_i in zip(line_saved, line_in):
        #     if x_s != float(x_i):
        #         same = False
    else: same = False

    if not same:
        print('ERROR: in_data_file_name does not match saved filtration.')
        print('Try running build_and_save_filtration()?')
        sys.exit()

def check_overwrite(out_file_name):
    os.chdir('output')
    if os.path.exists(out_file_name):
        overwrite = raw_input(out_file_name + " already exists. Overwrite? (y/n)\n")
        if overwrite == "y":
            pass
        else:
            print 'goodbye'
            sys.exit()
    os.chdir('..')

def make_filtration_movie(
        in_file_name,
        out_file_name,
        parameter_set,
        color_scheme='none',    # as of now, 'none', 'highlight new', or 'birth_time gradient'
        camera_angle=(135, 55), # for 3D mode. [azimuthal, elevation]
        alpha = 1,              # opacity (float, 0...1 : transparent...opaque)
        dpi=150,                # dots   per inch (resolution)
        max_frames = None,      # cut off frame (for testing or when only interested in the beginning of a movie)
        hide_1simplexes=False,  # i need to find a way to optimize the plotting of 1-simplexes(lines) 3D plotting, as of now they slow mayavi significantly.
        save_frames=False,      # save frames to /frames/ dir
        framerate=1             # number of frames per second. for a constant max_frames, higher framerate will make a shorter movie.
    ):

    check_saved_filtration(in_file_name)
    check_overwrite(out_file_name)
    start_time = time.time()
    title_block_info = [in_file_name, out_file_name, parameter_set, color_scheme, camera_angle, alpha, dpi, max_frames, hide_1simplexes]
    FiltrationPlotter.make_movie(out_file_name, title_block_info, color_scheme, alpha, dpi, framerate, camera_angle, hide_1simplexes, save_frames)
    print("make_filtration_movie() time elapsed: %d seconds \n" % (time.time() - start_time))


def make_persistence_diagram(
        in_file_name,
        out_file_name,
        parameter_set
    ):

    check_saved_filtration(in_file_name)
    check_overwrite(out_file_name)
    start_time = time.time()
    title_block_info = [in_file_name, out_file_name, parameter_set]
    PersistencePlotter.make_figure(title_block_info, out_file_name)
    print("make_persistence_diagram() time elapsed: %d seconds \n" % (time.time() - start_time))
