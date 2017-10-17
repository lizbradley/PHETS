import sys
import numpy as np
from config import WAV_SAMPLE_RATE



def embed(
		signal,
		tau,
		m,
		time_units='samples',
		crop=None,
		normalize=False,
		normalize_crop=False,
		ds_rate=1
):
	if normalize: signal = np.true_divide(signal, np.max(np.abs(signal)))

	signal = signal[::ds_rate]

	if crop is not None:
		if time_units == 'samples':
			pass
		elif time_units == 'seconds':
			crop = (np.array(crop) * WAV_SAMPLE_RATE).astype(int)
		else:
			print 'ERROR: invalid time_units'
			sys.exit()
		signal = signal[crop[0] : crop[1]]

	if time_units == 'seconds': tau = int(tau * WAV_SAMPLE_RATE)

	if normalize_crop: signal = np.true_divide(signal, np.max(np.abs(signal)))



	end = len(signal) - (tau * (m - 1)) - 1
	traj = []

	for i in range(end):
		pt = []
		for j in range(m):
			pt.append(signal[i + (j * tau)])
		traj.append(pt)

	return np.asarray(traj)










if __name__ == '__main__':
	pass