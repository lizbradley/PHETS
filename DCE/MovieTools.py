import sys
import os
import math
import shutil

from Tools import get_fund_freq, plot_power_spectrum

WAV_SAMPLE_RATE = 44100

def prep_save_worms_double():
	# ans = raw_input("Old worms will be deleted. OK? (y/n) \n")
	ans = 'y'
	if ans == 'y':
		dir_a = 'output/DCE/saved_worms/double/a/'
		for f in os.listdir(dir_a):
			os.remove(dir_a + f)

		dir_b = 'output/DCE/saved_worms/double/b/'
		for f in os.listdir(dir_b):
			os.remove(dir_b + f)

		open('output/DCE/saved_worms/double/a/WORM_INFO.txt', 'w').close()
		open('output/DCE/saved_worms/double/b/WORM_INFO.txt', 'w').close()
	else:
		sys.exit()


def save_worms_double(filename1, filename2, i, tau1, tau2, crop_1, crop_2):

	out_filename1 = 'output/DCE/saved_worms/double/a/' + filename1.split('/')[-1]
	out_filename2 = 'output/DCE/saved_worms/double/b/' + filename2.split('/')[-1]

	shutil.copyfile('DCE/temp_data/embedded_coords_comp1.txt', out_filename1)
	shutil.copyfile('DCE/temp_data/embedded_coords_comp2.txt', out_filename2)

	info_file_1 = open('output/DCE/saved_worms/double/a/WORM_INFO.txt', 'a')
	info_file_1.write('{:d}: tau = {} (samples), embed_crop = [{:.5f}, {:.5f}] (seconds) \n'.format(i, tau1, crop_1[0], crop_1[1]))
	info_file_1.close()

	info_file_1 = open('output/DCE/saved_worms/double/b/WORM_INFO.txt', 'a')
	info_file_1.write('{:d}: tau = {} (samples), embed_crop = [{:.5f}, {:.5f}] (seconds) \n'.format(i, tau2, crop_2[0], crop_2[1]))
	info_file_1.close()


def prep_save_worms_single():
	# ans = raw_input("Old worms will be deleted. OK? (y/n) \n")
	ans = 'y'
	if ans == 'y':
		dir = 'output/DCE/saved_worms/single/'
		for f in os.listdir(dir):
			os.remove(dir + f)

		open('output/DCE/saved_worms/single/WORM_INFO.txt', 'w').close()
	else:
		sys.exit()


def save_worms_single(filename, i, tau, embed_crop):

	out_filename = 'output/DCE/saved_worms/single/' + str(i) + '-' + filename.split('/')[-1]
	shutil.copyfile('DCE/temp_data/embedded_coords.txt', out_filename)
	info_file_1 = open('output/DCE/saved_worms/single/WORM_INFO.txt', 'a')
	info_file_1.write('{:d}: tau = {} (samples), embed_crop = [{:.5f}, {:.5f}] \n'.format(i, tau, embed_crop[0], embed_crop[1]))
	info_file_1.close()


def remove_old_frames():
	dir = 'DCE/frames/'
	for f in os.listdir(dir):
		if f.endswith(".png"):
			os.remove(dir + f)



def frames_to_movie(out_file_name, framerate=2):
	print 'building movie...'
	if os.path.exists(out_file_name):
		overwrite = raw_input(out_file_name + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			sys.exit()

	in_str = ('ffmpeg -y -framerate %i ' % framerate) + '-i DCE/frames/frame%03d.png'
	out_str = (' -r %d ' % 24) + out_file_name
	os.system(in_str + out_str)
	print os.getcwd() + ('\\' if os.name == 'nt' else '/') + out_file_name





