import numpy as np
from ROC import PRF_vs_FFT
from config import default_filtration_params as filt_params
from config import WAV_SAMPLE_RATE

test = 2


# TODO worm_length override
# TODO third plot sig panel: zoom window
# TODO time_units
# TODO FFT ROC accuracy vs FT_bins (logspace)
# TODO ROC color



if test == 1:


	filt_params.update(
		{
			'max_filtration_param': -5,
			'num_divisions': 10,
			'use_cliques': True,
		}
	)

	PRF_vs_FFT(
		'datasets/time_series/clarinet/sustained/high_quality/40-clarinet-HQ.txt',
		'datasets/time_series/viol/40-viol.txt',
		'output/ROC/test_1.png',

		'clarinet',
		'viol',

		crop_1=(100000, 170000),
		crop_2=(50000, 120000),

		tau=32,  # samples
		m=2,

		window_length=2000,
		# window_length=(1000, 8000, 24000),
		num_windows=15,
		num_landmarks=30,
		FT_bins=50,
		k=(0, 100, .1),		# min, max, int
		load_saved_filts=False,
		normalize_volume=True

	)


def sec_to_samp(crop):
	return (np.array(crop) * WAV_SAMPLE_RATE).astype(int)

if test == 2:




	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	PRF_vs_FFT(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		'output/ROC/test_1_k10_numdiv30_mfp8_norm.png',

		'clarinet',
		'viol',



		crop_1=sec_to_samp((2, 4)),
		crop_2=sec_to_samp((1, 3)),

		tau=311,  # samples
		m=2,

		window_length=2205,
		# window_length=(1000, 8000, 24000),
		num_windows=30,
		num_landmarks=55,
		FT_bins=50,
		k=(0, 2, .01),		# min, max, int
		load_saved_filts=True,
		normalize_volume=True

	)