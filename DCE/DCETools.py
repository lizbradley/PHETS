import sys
import os
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as pyplot

def wav_to_txt(wav_file_name, output_file_name, crop=(0, 1)):
    print "converting wav to txt..."
    sampFreq, sig = wavfile.read(wav_file_name)
    bounds = len(sig) * np.array(crop)
    sig = sig[int(bounds[0]):int(bounds[1])]
    np.savetxt(output_file_name, sig)



def embed(input_file_name, output_file_name, embed_crop, tau, m, wav_sample_rate,
          ds_rate=1,
          channel=0):

    # print 'embedding...'
    input_file = open(input_file_name, "r")
    output_file = open(output_file_name, "w")
    output_file.truncate(0)
    lines = input_file.read().split("\n")

    worm_length_sec = len(lines) / wav_sample_rate
    embed_crop_norm = [float(t) / worm_length_sec for t in embed_crop]

    bounds = len(lines) * np.array(embed_crop_norm)
    lines = lines[int(bounds[0]): int(bounds[1]) : ds_rate]

    series = []
    for line in lines:
        if line != "":
            channels = [x for x in line.split(" ") if x != ""]
            series.append(float(channels[channel]))
    end = len(lines) - (tau*(m - 1)) - 1
    for i in xrange(end):
        for j in xrange(m):
            output_file.write("%f " % series[i + (j*tau)])
        if i < end:
            output_file.write("\n")
    input_file.close()
    output_file.close()

def rename_files():
    os.chdir('input/viol_data')
    [os.rename(f, f.replace('-consolidated', '')) for f in os.listdir('.') if f.endswith('.wav') or f.endswith('.txt')]

def rename_files_shift_index():
    os.chdir('input/viol_data')
    for f in os.listdir('.'):
        i_in = int(f.split('-')[0])
        if i_in > 57:
            base = f.split('-')[1]
            i_out = i_in + 1
            os.rename(f, "temp{:02d}-{}".format(i_out, base))

    for f in os.listdir('.'):
        if 'temp' in f:
            os.rename(f, f.replace('temp', ''))



def batch_wav_to_txt(dir_name):
    os.chdir(dir_name)
    [wav_to_txt(f, f.replace('.wav', '.txt')) for f in os.listdir('.') if f.endswith('.wav')]



def get_fund_freq(filename, expected, window=(1, 2), tol=10):
    samp_freq = 44100.
    window_sec = window
    window = np.array(window) * samp_freq
    sig = np.loadtxt(filename)

    # sig_lowpass = butter_lowpass_filter(sig, 20, samp_freq, order=1)

    sig_crop = sig[int(window[0]): int(window[1])]

    window_len_sec = window_sec[1] - window_sec[0]
    spec_prec = int(100000 / (samp_freq * window_len_sec))  # hz ?

    FFT_x = scipy.fftpack.fftfreq(sig_crop.size * spec_prec, d=1/samp_freq)

    FFT = scipy.fftpack.fft(sig_crop, len(sig_crop) * spec_prec)
    FFT = 20 * scipy.log10(scipy.absolute(FFT)) # convert to db
    FFT_pos = FFT[1:len(FFT)/2]
    FFT_neg = FFT[(len(FFT)/2) + 1:]

    spec = FFT_pos + FFT_neg[::-1]
    spec_x = FFT_x[1:len(FFT_x)/2]

    freq_window_idx = [i for i, x in enumerate(spec_x) if np.abs(expected - x) < tol]
    if len(freq_window_idx) ==0:
        print("ERROR: No fundamental frequency found. Increase 'tol'.")
        sys.exit()
    freq_window_freq = spec_x[freq_window_idx]
    freq_window_amp = spec[freq_window_idx]
    # plot_spec(spec, spec_x, 'lowpass.png')

    max_idx = np.argmax(freq_window_amp)
    fund = freq_window_freq[max_idx]
    return fund

# from scipy.interpolate import interp1d

def plot_spec(spec_x, spec, out_file):
    fig = pyplot.figure()
    plt = fig.add_subplot(111)
    plt.set_xscale('log')
    plt.set_xlim([20, 20000])
    plt.plot(spec, spec_x, c='k', lw=.1)
    plt.set_xlabel('frequency (Hz)')
    plt.grid()


    pyplot.savefig(out_file)
    pyplot.close(fig)


def plot_power_spectrum(sig, out_file, crop=(1,2)):
    from DCEPlotter import plot_waveform
    samp_freq = 44100.

    if crop != 'none':
        window = np.array(crop) * samp_freq
        sig_crop = sig[int(window[0]):int(window[1])]
    else:
        sig_crop = sig
    FFT = scipy.fftpack.fft(sig_crop)
    FFT = 20 * scipy.log10(scipy.absolute(FFT)) # convert to db
    FFT_x = scipy.fftpack.fftfreq(len(FFT), d=1 / samp_freq)

    fig, subplots = pyplot.subplots(2, figsize=(6, 3), dpi=300, tight_layout=True)


    FFT_pos = FFT[1:len(FFT)/2]
    FFT_neg = FFT[(len(FFT)/2) + 1:]
    spec = FFT_pos + FFT_neg[::-1]

    # TODO: show grid, more ticks

    subplots[0].set_xscale('log')
    subplots[0].set_xlim([20, 20000])
    subplots[0].plot(FFT_x[1:len(FFT_x)/2], spec, c='k', lw=.1)
    subplots[0].set_xlabel('frequency (Hz)')

    plot_waveform(subplots[1], sig, embed_crop=crop)


    pyplot.savefig(out_file)
    pyplot.close(fig)


if __name__ == '__main__':
    print os.getcwd()
    # rename_files_shift_index()
    # batch_wav_to_txt('C:\Users\PROGRAMMING\Documents\CU_research\piano_data\C134C')
    # batch_wav_to_txt('input/viol_data')
    # get_fund_freq('input/viol_data/01-viol.txt', window=(1, 2))
    # get_fund_freq('input/piano_data/C134C/24-C134C.txt', window=(1, 2))