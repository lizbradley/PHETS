import sys
from Utilities import pwd
import numpy as np



WAV_SAMPLE_RATE = 44100



def embed_v1(
		signal, output_file_name,
		embed_crop,		# sec
		tau,			# sec
		m,
		ds_rate=1,
		channel=0
):

	if embed_crop:
		embed_crop_samp = np.array(embed_crop) * WAV_SAMPLE_RATE
		signal = signal[int(embed_crop_samp[0]):int(embed_crop_samp[1])]

	tau_samp = int(tau * WAV_SAMPLE_RATE)
	end = len(signal) - (tau_samp * (m - 1)) - 1

	output_file = open(output_file_name, "w")
	output_file.truncate(0)


	for i in xrange(end):
		for j in xrange(m):
			output_file.write("%f " % signal[i + (j*tau_samp)])
		if i < end:
			output_file.write("\n")
	output_file.close()



def embed(
		signal,
		tau,
		m,
		time_units='samples',
		embed_crop=None,
		ds_rate=1,
		channel=0,
		normalize=False
):
	if normalize:
		signal = np.true_divide(signal, np.max(np.abs(signal)))


	if embed_crop:
		if time_units == 'samples':
			pass
		elif time_units == 'seconds':
			embed_crop = (np.array(embed_crop) * WAV_SAMPLE_RATE).astype(int)
		else:
			print 'ERROR: invalid time_units'
			sys.exit()
		signal = signal[embed_crop[0] : embed_crop[1]]

	if time_units == 'seconds':
		tau = int(tau * WAV_SAMPLE_RATE)



	end = len(signal) - (tau * (m - 1)) - 1
	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(signal[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)








# from scipy.interpolate import interp1d

#
# def auto_embed(
# 		filename,
#
# 		crop='auto',
# 		auto_crop_length=.3,
#
# 		tau='auto ideal',
# 		tau_T = np.pi,
# 		note_index=None
# 		):
#
# 	if isinstance(crop, basestring):
# 		if crop == 'auto':
# 			crop = np.array(auto_crop(np.loadtxt(filename), auto_crop_length)) / float(WAV_SAMPLE_RATE)
# 		else:
# 			print "ERROR: embed_crop_1 not recognized. Use 'auto' or explicit (seconds)."
# 			sys.exit()
#
#
#
# 	if isinstance(tau, basestring):
# 		if not note_index: print "ERROR: note index required for tau='auto ideal' and tau='auto detect'"
#
# 		ideal_freq = math.pow(2, (40 - float(note_index)) / 12) * 440  # Hz, descending index
#
# 		if tau == 'auto detect': f = get_fund_freq(filename, ideal_freq, window=crop)
#
# 		if tau == 'auto ideal': f = ideal_freq
#
# 		else:
# 			print 'ERROR: tau not recognized.'
# 			sys.exit()
#
# 	embed(filename, 'DCE/temp_data/embedded_coords_comp1.txt', crop_1, tau_1, m, ds_rate=ds_rate)
#


if __name__ == '__main__':
	pass