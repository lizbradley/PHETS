import sys, time
from ROC import PRF_vs_FFT_v2
from ROC.ROCs import classifier_ROC
from Tools import sec_to_samp
from config import default_filtration_params as filt_params


# TODO worm_length override
# TODO third plot sig panel: zoom window
# TODO time_units
# TODO FFT ROC accuracy vs FT_bins (logspace)


set_test = 22




if len(sys.argv) > 1: test = int(sys.argv[1])
else: test = set_test
print 'running test %d...' % test
start_time = time.time()


def out_fname():
	return 'output/ROC/test_{}.png'.format(test)




if test == 6:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	PRF_vs_FFT_v2(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		out_fname(),

		'clarinet',
		'viol',

		crop_1=sec_to_samp((2, 4)),
		crop_2=sec_to_samp((1, 3)),

		tau=311,  # samples
		m=2,

		window_length=2000,
		# window_length=(1000, 8000, 24000),
		num_windows=30,
		num_landmarks=55,
		FT_bins=50,
		k=(0, 2, .01),		# min, max, step
		load_saved_filts=True,
		normalize_volume=True

	)


if test == 7:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 10,
			'use_cliques': True,
		}
	)

	PRF_vs_FFT_v2(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		# out_fname(),
		'output/ROC/test7tau311',

		'clarinet',
		'viol',

		crop_1=sec_to_samp((2, 4)),
		crop_2=sec_to_samp((1, 3)),

		tau=311,  # samples
		m=2,

		window_length=1000,
		# window_length=(1000, 8000, 24000),
		num_windows=15,
		num_landmarks=55,
		FT_bins=50,
		k=(0, 10, .001),		# min, max, step
		load_saved_filts=False,
		normalize_volume=True

	)



if test == 20:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	PRF_vs_FFT_v2(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		out_fname(),

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
		k=(0, 5, .01),		# min, max, int
		load_saved_filts=True,
		normalize_volume=True

	)



if test == 21:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	classifier_ROC(
		'datasets/time_series/Clarinet/40-clarinet.txt',
		'datasets/time_series/viol/40-viol.txt',
		out_fname(),

		'clarinet',
		'viol',

		crop_1=sec_to_samp((2, 4)),
		crop_2=sec_to_samp((1, 3)),

		tau=311,  # samples
		m=2,

		window_length=2205,
		# window_length=(1000, 8000, 24000),
		num_windows=10,
		num_landmarks=55,
		k=(0, 5, .01),		# min, max, int
		load_saved_filts=False,
		normalize_volume=True,
		save_samps=False

	)


if test == 22:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	classifier_ROC(
		'ClassicBifurcationData/NewHopf_anp01.txt',
		'ClassicBifurcationData/NewHopf_a0.txt',
		out_fname(),

		'a = NewHopf -0.01',
		'a = NewHopf 0.00',

		crop_1=(100, 9000),
		crop_2=(100, 9000),

		tau=311,  # samples
		m=2,

		window_length=2205,
		# window_length=(1000, 8000, 24000),
		num_windows=10,
		num_landmarks=55,
		k=(0, 5, .01),		# min, max, int
		load_saved_filts=True,
		normalize_volume=True,
		save_samps=True

	)

if test == 23:

	filt_params.update(
		{
			'max_filtration_param': -8,
			'num_divisions': 30,
			'use_cliques': True,
		}
	)

	classifier_ROC(
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau11.txt',
		'datasets/trajectories/L63_x_m2/L63_x_m2_tau25.txt',
		out_fname(),

		'a = -0.01',
		'a = -0.5',

		crop_1=(100, 9000),
		crop_2=(100, 9000),

		tau=311,  # samples
		m=2,

		window_length=2205,
		# window_length=(1000, 8000, 24000),
		num_windows=10,
		num_landmarks=55,
		k=(0, 5, .01),		# min, max, int
		load_saved_filts=False,
		normalize_volume=True,
		save_samps=True

	)