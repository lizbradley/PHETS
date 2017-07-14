Reference
=========

.. module:: PHETS.PH
   :synopsis: Computation and visualization of persistent homology.



.. class:: Filtration(sig, params[, filename='none'])

   :param sig: Input trajectory as numpy array or path to .txt file.
   :type sig: ndarray or str
   :param dict params: Options for computation of filtration.
   :param str filename: If sig is an ndarray, a filename may be provided for labelling in visualizations.



.. function:: make_movie(filtration, out_filename[, color_scheme='none', camera_angle=(135, 55), alpha=1, dpi=150, save_frames=False, framerate=1])

   Make filtration movie.

   :param Filtration filtration:
   :param str out_filename: Path/filename for movie. Should end in ".mp4" or other video format.
   :param str color_scheme: 'none', 'highlight new', or 'birth_time gradient'
   :param tuple camera_angle: for 3D mode. (azimuthal, elevation) in degrees.
   :param float alpha: opacity of simplexes. 0...1 : transparent...opaque
   :param int dpi: dots per inch (resolution)
   :param bool save_frames: save frames to PH/frames/frame*.png for debugging
   :param int framerate: frames per second



.. function:: make_PD(filtration, out_filename)

   Plot persistence diagram.

   :param Filtration filtration:
   :param str out_filename: Image path and filename. Should end in ".png" or other image format.



.. function:: make_PRF_plot(filt, out_filename[, PRF_res=50])

   Plot persistent rank function.

   :param Filtration filtration:
   :param str out_filename: Image path and filename. Should end in ".png" or other image format.
   :param int PRF_res: number of divisions per epsilon axis



.. todo:: a function for a single frame of the filtration (i.e. fixed epsilon) as an image. (For now, a single frame movie can be used.)






.. module:: PHETS.DCE.Movies
   :synopsis: Generate and visualize delay coordinate embeddings time series

.. function:: slide_window(in_filename, out_filename, window_size=.5, step_size=.1, tau=10, ds_rate=1, max_frames=0, save_trajectories=True, save_movie=True)

   Show embedding of in_filename with window start point varied over time.

   :param float window_size: seconds
   :param step_size: window start point step size (seconds)
   :param float tau: seconds ?



.. function:: vary_tau(in_filename, out_filename, tau_lims=(1, 15), tau_inc=1, embed_crop=(1, 2), ds_rate=1, save_trajectories=True, save_movie=True, m=2)

   Show embedding of in_filename with tau varied over time.

   :param str in_filename: Path/filename for text file time series.
   :param str out_filename: Path/filename for movie. Should end in ".mp4" or other video format.
   :param tuple tau_lims: tau range (seconds)
   :param int tau_inc: tau stepsize (seconds)
   :param tuple embed_crop: Limits for window from input time series (seconds)
   :param int ds_rate: time series downsample rate
   :param bool save_trajectories: save embeddings to text files in output/DCE/trajectories
   :param bool save_movie: If False, no movie will be created. Useful for saving embeddings quickly.
   :param int m: target embedding dimension



.. function:: compare_vary_tau(in_filename_1, in_filename_2, out_filename, tau_lims, tau_inc=1,	embed_crop=(1, 2), ds_rate=1, m=2, save_trajectories=True,	save_movie=True)

   Like vary_tau(), but shows embeddings for two time series side by side.



.. function:: compare_multi(dir1, dir1_base, dir2, dir2_base, out_filename, i_lims=(1, 89), embed_crop_1='auto', embed_crop_2='auto', auto_crop_length=.3, tau_1='auto ideal', tau_2='auto ideal', tau_T=1/np.pi, save_trajectories=True, save_movie=True, normalize_volume=True, waveform_zoom=None, ds_rate=1, dpi=200, m=2)

   Takes two directories of (eg one with piano notes, another with range of viol notes), and generates a movie over a range note indexes (pitch). Tau and crop may be set explicity or automatically.

   :param str dir1: Path of first directory to be iterated over
   :param str dir1_base: Base filename for files in dir1
   :param str dir2:
   :param str dir2_base:
   :param str out_filename:
   :param tuple i_lims: (start, stop) index. Default is (1, 89).
   :param embed_crop_1: (start, stop) in seconds or 'auto'
   :type embed_crop_1: tuple or str
   :param embed_crop_2:
   :type embed_crop_2: tuple or str
   :param float auto_crop_length=.3: seconds
   :param str tau_1: explicit (seconds) or 'auto detect' or 'auto ideal'
   :param str tau_2:
   :param float tau_T: For use with auto tau: tau = period * tau_T
   :param bool save_trajectories:
   :param bool save_movie:
   :param bool normalize_volume:
   :param waveform_zoom:
   :param int ds_rate:
   :param int dpi:



.. todo:: function for plotting embeddings without varying a parameter or input, as an image. (For now, a single frame movies can be used.)




.. module:: PHETS.PRFCompare
   :synopsis: Generation, statistical analysis, and visualization of sets of persistent rank functions.


.. function:: plot_dists_vs_ref(dir, base_filename, fname_format, out_filename, filt_params, i_ref=15, i_arr=np.arange(10, 20, 1), weight_func=lambda i, j: 1, metric='L2', dist_scale='none', PRF_res=50, load_saved_PRFs=False, see_samples=5)

   Takes range of time-series files and a reference file. Generates PRF for each, and finds distances to reference PRF, plots distance vs index.

   :param str dir: input directory
   :param str base_filename: input base filename
   :param str fname_format: input filename format: 'base i or 'i base'
   :param str out_filename: output filename
   :param filt_params:
   :param int i_ref:
   :param arr i_arr:
   :param lambda weight_func: Default is lambda i, j: 1
   :param str metric: 'L1' (abs) or 'L2' (euclidean). Default is 'L2'.
   :param str dist_scale: 'none', 'a', or 'a + b'. Default is 'none'.
   :param int PRF_res: number of divisions used for PRF. Default is 50.
   :param bool load_saved_PRFs: reuse previously computed PRF set. Default is False.
   :param int see_samples: interval to generate PRF plots, PDs, and filtration movies when generating PRF set. 0 is none, 1 is all samples, 2 is every other sample, etc.


.. function:: plot_dists_vs_mean(filename_1, filename_2, out_filename, filt_params, load_saved_PRFs=False, time_units='seconds', crop_1='auto', crop_2='auto', auto_crop_length=.3, window_size=.05, num_windows=10, mean_samp_num=5, tau_1=.001, tau_2=.001, tau_T=np.pi, note_index=None, normalize_volume=True, PRF_res=50, dist_scale='none', metric='L2', weight_func=lambda i, j: 1, see_samples=5)

   Takes two time-series or 2D trajectory files. For each input, slices each into a number of windows. If inputs are time-series, embeds each window. Generates PRF for each window. selects subset of window PRFs, computes their mean, plots distance to mean PRF vs time.

   :param str filename_1:
   :param str filename_2:
   :param str out_filename:
   :param dict filt_params:
   :param bool load_saved_PRFs:
   :param str time_units:
   :param crop_1:
   :param crop_2:
   :type crop_1: str or tuple
   :type crop_2: str or tuple
   :param float auto_crop_length:
   :param int num_windows: per file
   :param int mean_samp_num: per file
   :param tau_1:
   :param tau_2:
   :type tau_1: str or float
   :type tau_2: str or float
   :param float tau_T:
   :param int note_index:
   :param bool normalize_volume:
   :param int PRF_res: number of divisions used for PRF
   :param str dist_scale: 'none', 'a', or 'a + b'
   :param str metric: 'L1' (abs) or 'L2' (euclidean)
   :param lambda weight_func:
   :param int see_samples:



.. function:: plot_clusters(*args, **kwargs)

   See plot_dists_vs_mean for call signature.



.. function:: plot_variances(filename, out_filename, filt_params, vary_param_1, vary_param_2, load_saved_PRFs=False, time_units='seconds', crop=(100, 1100), auto_crop_length=.3, window_size=1000, num_windows=5, tau=.001, tau_T=np.pi, note_index=None, normalize_volume=True, PRF_res=50, dist_scale='none', metric='L2', weight_func=lambda i, j: 1, see_samples=5)

   :param str filename:
   :param str out_filename:
   :param dict filt_params:
   :param  vary_param_1: filtration parameter to vary over x axis
   :param vary_param_2: filtration parameter to vary over line colors
   :type vary_param_1: (str, tuple)
   :type vary_param_2: None or (str, tuple)
   :param bool load_saved_PRFs: reuse saved
   :param str time_units='seconds':
   :param tuple crop=(100, 1100): (start, stop) in time units
   :param int num_windows: Number of windows to select from crop, evenly spaced. Window length is  chosen with the 'worm_length' filtration parameter. Windows may or may not overlap
   :param tau: time units
   :type tau: int or float
   :param bool normalize_volume: normalize volume (per crop)
   :param bool normalize_sub_volume: normalize volume (per window) [coming soon]
   :param int PRF_res:
   :param str dist_scale: 'none', 'a', or 'a + b'
   :param str metric: 'L1' (abs) or 'L2' (euclidean)
   :param lambda weight_func:
   :param bool see_samples:



