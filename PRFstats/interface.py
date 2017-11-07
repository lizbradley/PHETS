import sys

import numpy as np

from PRFstats.data import roc_data, fetch_filts
from PRFstats.plots import dists_vs_means_fig, clusters_fig
from data import L2Classifier, prf_dists_compare
from plots import dual_roc_fig, samples
from utilities import clear_old_files


def plot_dists_vs_ref(
		dir, base_filename,
		fname_format,
		out_filename,
		filt_params,

		i_ref=15,
		i_arr=np.arange(10, 20, 1),

		weight_func=lambda i, j: 1,

		metric='L2',  # 'L1' (abs) or 'L2' (euclidean)
		dist_scale='none',  # 'none', 'a', or 'a + b'


		load_saved_PRFs=False,

		see_samples=5,

):
	""" plots distance from reference rank function over a range of trajectories input files"""

	def main():
		clear_old_files('output/PRFCompare/ref/see_samples/', see_samples)

		if load_saved_PRFs:
			print 'WARNING: loading saved filtration'
			funcs = np.load('PRFCompare/funcs.npy')
			ref_func = np.load('PRFCompare/ref_func.npy')
		else:
			funcs = get_PRFs()  # also makes PDs and movies
			ref_filt = Filtration(get_in_filename(i_ref), filt_params)
			if see_samples: show_samples(ref_filt, i_ref, ref=True)
			ref_func = ref_filt.get_PRF()
			np.save('PRFCompare/funcs.npy', funcs)
			np.save('PRFCompare/ref_func.npy', ref_func)

		make_PRF_plot(ref_func, 'output/PRFCompare/ref/PRF_REFERENCE.png',
		              params=filt_params, in_filename='REF')

		funcs_z = funcs[:, 2]
		ref_func_z = ref_func[2]
		dists = get_dists_from_ref(funcs_z, ref_func_z, metric, dist_scale)
		plot_distances(i_ref, i_arr, dists, out_filename)


	def plot_distances(i_ref, i_arr, dists, out_filename):
		fig = plt.figure(figsize=(10, 5))
		ax = fig.add_subplot(111)
		ax.plot(i_arr, dists)
		ax.axvline(x=i_ref, linestyle='--', color='k')
		ax.set_xlabel('$tau \quad (samples)$')
		# ax.set_ylabel('$distance \quad ({\epsilon}^2 \; \# \; holes)$')
		ax.set_ylabel('$distance$')
		ax.xaxis.set_ticks(i_arr[::2])
		ax.grid()
		ax.set_ylim(bottom=0)
		title = ax.set_title(base_filename + ' PRF distances')
		title.set_position([.5, 1.05])
		plt.savefig(out_filename)
		plt.close(fig)


	def get_in_filename(i):
		if fname_format == 'i base':
			filename = '{}/{}{}'.format(dir, i, base_filename)
		elif fname_format == 'base i':
			filename = '{}/{}{}.txt'.format(dir, base_filename, i)
		else:
			print "ERROR: invalid fname_format. Valid options: 'i base', 'base i'"
			sys.exit()
		return filename


	def show_samples(filt, i, ref=False):
		os.chdir('..')
		base_name = base_filename.split('/')[-1].split('.')[0]
		comp_name = '{:s}_{:d}_'.format(base_name, i)
		if ref: comp_name += '_REFERENCE_'
		PD_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'PD.png'
		movie_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'movie.mp4'
		PRF_filename = 'output/PRFCompare/ref/see_samples/' + comp_name + 'PRF.png'

		make_PD(filt, PD_filename)
		make_PRF_plot(filt, PRF_filename)
		make_movie(filt, movie_filename)
		os.chdir('PRFCompare')


	def get_PRFs():
		funcs = []
		for i in i_arr:
			filename = get_in_filename(i)
			print '\n=================================================='
			print filename
			print '==================================================\n'
			filt = Filtration(filename, filt_params)
			func = filt.get_PRF()
			funcs.append(func)

			if see_samples:
				if i % see_samples == 0:
					show_samples(filt, i)

		return np.asarray(funcs)


	main()

def plot_dists_vs_means(
		traj1,
		traj2,
		out_filename,
		filt_params,
		time_units='samples',
		metric='L2',
		dist_scale='none',              # 'none', 'a', or 'a + b'
		weight_func=lambda i, j: 1,
		see_samples=5,
		load_saved_filts=False,
		quiet=True
	):

	# TODO: weight_func, see_samples, time_units, unit test

	filts1, filts2 = fetch_filts(
		traj1, traj2, filt_params,
		load_saved_filts, quiet
	)

	prfs1 = [f.get_PRF(silent=quiet, new_format=True) for f in filts1]
	prfs2 = [f.get_PRF(silent=quiet, new_format=True) for f in filts2]

	refs, dists = prf_dists_compare(prfs1, prfs2, metric, dist_scale)

	dists_vs_means_fig(refs, dists, traj1, traj2, time_units, out_filename)


def plot_clusters(
		traj1,
		traj2,
		out_filename,
		filt_params,
		metric='L2',
		dist_scale='none',              # 'none', 'a', or 'a + b'
		weight_func=lambda i, j: 1,
		see_samples=5,
		load_saved_filts=False,
		quiet=True
):

	# TODO: weight_func, see_samples, unit test


	filts1, filts2 = fetch_filts(
		traj1, traj2, filt_params,
	    load_saved_filts, quiet
	)

	prfs1 = [f.get_PRF(silent=quiet, new_format=True) for f in filts1]
	prfs2 = [f.get_PRF(silent=quiet, new_format=True) for f in filts2]

	refs, dists = prf_dists_compare(prfs1, prfs2, metric, dist_scale)

	clusters_fig(dists, filt_params, traj1.name, traj2.name,out_filename)


def L2ROCs(
		traj1, traj2,
		label1, label2,
		out_fname,
		filt_params,
		k,
		load_saved_filts=False,
		see_samples=0,
		quiet=True,
		vary_param=None     # ('param', (100, 150, 200))
):
	# TODO: add weight function


	filts1_v, filts2_v = fetch_filts(     # filts varied over vary_param
		traj1, traj2, filt_params,
		load_saved_filts, quiet,
		vary_param_1=vary_param
	)

	if vary_param is None:
		filts1_v, filts2_v = [filts1_v], [filts2_v]
	data = []

	for filts1, filts2 in zip(filts1_v, filts2_v):

		prfs1 = [f.get_PRF(silent=quiet, new_format=True) for f in filts1]
		prfs2 = [f.get_PRF(silent=quiet, new_format=True) for f in filts2]

		train1, test1 = prfs1[1::2], prfs1[::2]
		train2, test2 = prfs2[1::2], prfs2[::2]

		print 'training classifiers...'
		clf1 = L2Classifier(train1)
		clf2 = L2Classifier(train2)

		print 'running tests...'
		k_arr = np.arange(*k)
		roc1 = roc_data(clf1, test1, test2, k_arr)
		roc2 = roc_data(clf2, test2, test1, k_arr)

		data.append([roc1, roc2])

	dual_roc_fig(data, k, label1, label2, out_fname, vary_param)

	if see_samples:
		dir = 'output/PRFstats/samples'
		clear_old_files(dir, see_samples)

		if vary_param is None:
			filts1_v, filts2_v = filts1_v[0], filts2_v[0]

		samples(filts1_v, see_samples, dir, vary_param)
		samples(filts2_v, see_samples, dir, vary_param)

	return data


# TODO: plot_variance
