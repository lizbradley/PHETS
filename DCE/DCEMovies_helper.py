import sys
import os
import math
import shutil

from DCETools import get_fund_freq
from DCETools import plot_power_spectrum

def prep_save_worms_double():
    ans = raw_input("Old worms will be deleted. OK? (y/n) \n")
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


def save_worms_double(filename1, filename2, i, tau1, tau2, embed_crop):

    out_filename1 = 'output/DCE/saved_worms/double/a/' + filename1.split('/')[-1]
    out_filename2 = 'output/DCE/saved_worms/double/b/' + filename2.split('/')[-1]

    shutil.copyfile('temp_data/embedded_coords_comp1.txt', out_filename1)
    shutil.copyfile('temp_data/embedded_coords_comp2.txt', out_filename2)

    info_file_1 = open('output/DCE/saved_worms/double/a/WORM_INFO.txt', 'a')
    info_file_1.write('{:d}: tau = {} (samples), embed_crop = [{:.5f}, {:.5f}] (seconds) \n'.format(i, tau1, embed_crop[0], embed_crop[1]))
    info_file_1.close()

    info_file_1 = open('output/DCE/saved_worms/double/b/WORM_INFO.txt', 'a')
    info_file_1.write('{:d}: tau = {} (samples), embed_crop = [{:.5f}, {:.5f}] (seconds) \n'.format(i, tau2, embed_crop[0], embed_crop[1]))
    info_file_1.close()


def prep_save_worms_single():
    ans = raw_input("Old worms will be deleted. OK? (y/n) \n")
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
    for f in os.listdir('DCE/frames'):
        if f.endswith(".png"):
            os.remove('DCE/frames/' + f)



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


def get_freq_info(f, tau_T, embed_len_sec):
    period = 1 / f
    tau_sec = period * tau_T
    sample_rate = 44100
    tau_samp = int(tau_sec * sample_rate)
    cycles = embed_len_sec / period

    return [tau_samp, tau_sec, period, cycles]





def get_info_ideal(tau_T, embed_crop, i, ds_rate, wav_sample_rate, filename1, filename2):
    # ideal_freq = math.pow(2, (i - 49)/12) * 440    # Hz, ascending index


    embed_len_sec = (abs(float(embed_crop[1] - float(embed_crop[0]))))
    num_samples = int(embed_len_sec * wav_sample_rate / ds_rate)

    # ideal_freq = math.pow(2, (i - 49)/12) * 440    # Hz, ascending index
    ideal_freq = math.pow(2, (40 - float(i)) / 12) * 440  # Hz, descending index
    # f1 = get_fund_freq(filename1, ideal_freq, embed_crop)
    # f2 = get_fund_freq(filename2, ideal_freq, embed_crop)

    f_info = get_freq_info(ideal_freq, tau_T, embed_len_sec)
    # f_info_1 = get_freq_info(f1, tau_T, embed_len_sec)
    # f_info_2 = get_freq_info(f2, tau_T, embed_len_sec)

    info_main = [
        ['tau (samples)', '{:d}'.format(f_info[0])],
        ['tau (sec)', '{:.4f}'.format(f_info[1])],
        ['period (sec)', '{:.4f}'.format(f_info[2])],
        ['cycles', '{:.4f}'.format(f_info[3])],
        ['f (Hz) [ideal]', '{:.1f}'.format(ideal_freq)],
        ['tau/period', '{:.4f}'.format(f_info[1]/f_info[2])],
        ['num samples', '{:d}'.format(num_samples)],
        ['ds rate', '{:d}'.format(ds_rate)],
    ]

    info_1 = [filename1]

    info_2 = [filename2]

    info = {
        'tau': f_info[0],
        'tau_1':[],
        'tau_2':[],
        'title_main': info_main,
        'title_1':info_1,
        'title_2':info_2
    }

    return info


def get_info_real(tau_T, embed_crop, i, ds_rate, wav_sample_rate, filename1, filename2):
    embed_len_sec = (abs(float(embed_crop[1] - float(embed_crop[0]))))
    num_samples = embed_len_sec * wav_sample_rate / ds_rate

    # ideal_freq = math.pow(2, (i - 49)/12) * 440    # Hz, ascending index
    ideal_freq = math.pow(2, (40 - float(i)) / 12) * 440  # Hz, descending index


    f1 = get_fund_freq(filename1, ideal_freq, embed_crop)
    f2 = get_fund_freq(filename2, ideal_freq, embed_crop)

    f_info_1 = get_freq_info(f1, tau_T, embed_len_sec)
    f_info_2 = get_freq_info(f2, tau_T, embed_len_sec)


    info_main = [
        ['f (Hz) [ideal]', '{:.1f}'.format(ideal_freq)],
        ['num samples', '{:d}'.format(int(num_samples))],
        ['ds rate', '{:d}'.format(ds_rate)],
    ]

    info_1 = [
        filename1,
        ['f (Hz) [det]', '{:.1f}'.format(f1)],
        ['tau (samples)', '{:d}'.format(f_info_1[0])],
        ['tau (sec)', '{:.4f}'.format(f_info_1[1])],
        ['period (sec)', '{:.4f}'.format(f_info_1[2])],
        ['cycles', '{:.4f}'.format(f_info_1[3])],
        ['tau/period', '{:.4f}'.format(f_info_1[1]/f_info_2[2])]
    ]

    info_2 = [
        filename2,
        ['f (Hz) [det]', '{:.1f}'.format(f2)],
        ['tau (samples)', '{:d}'.format(f_info_2[0])],
        ['tau (sec)', '{:.4f}'.format(f_info_2[1])],
        ['period (sec)', '{:.4f}'.format(f_info_2[2])],
        ['cycles', '{:.4f}'.format(f_info_2[3])],
        ['tau/period', '{:.4f}'.format(f_info_2[1]/f_info_2[2])]
    ]

    info = {
        'tau_1':f_info_1[0],
        'tau_2':f_info_2[0],
        'title_main': info_main,
        'title_1':info_1,
        'title_2':info_2
    }

    return info

