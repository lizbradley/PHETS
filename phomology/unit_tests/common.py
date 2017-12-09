from signals import Trajectory
from phomology import Filtration
from config import default_filtration_params as dfp

filt_params = dfp.copy()


ellipse_traj = Trajectory('data/ellipse.txt')


euc_params = filt_params.copy()
euc_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
	})
euc_filt = Filtration(ellipse_traj, euc_params, save=False)


ham5_params = filt_params.copy()
filt_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_use_hamiltonion': 5,
	})
ham5_filt = Filtration(ellipse_traj, ham5_params, save=False)


doamp5_params = filt_params.copy()
doamp5_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_orientation_amplify': 5,
	})
doamp5_filt = Filtration(ellipse_traj, doamp5_params, save=False)


m2d10_params = filt_params.copy()
m2d10_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_use_hamiltonion': 0,
		'm2_d': 10,
	})
m2d10_filt = Filtration(ellipse_traj, m2d10_params, save=False)


dcov20_params = filt_params.copy()
dcov20_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		"d_cov": 20,

	})
dcov20_filt = Filtration(ellipse_traj, dcov20_params, save=False)


dcovn20_params = filt_params.copy()
dcovn20_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		"d_cov": -20,
	})
dcovn20_filt = Filtration(ellipse_traj, dcovn20_params, save=False)


gi_params = filt_params.copy()
gi_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'graph_induced': True
	})
gi_filt = Filtration(ellipse_traj, gi_params, save=False)


hamn10_params = filt_params.copy()
hamn10_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_use_hamiltonian': -10

	})
hamn10_filt = Filtration(ellipse_traj, hamn10_params, save=False)


hamn1_params = filt_params.copy()
hamn1_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_use_hamiltonian': -1

	})
hamn1_filt = Filtration(ellipse_traj, hamn1_params, save=False)


ham1_params = filt_params.copy()
ham1_params.update(
	{
		'ds_rate': 100,
		'max_filtration_param': -5,
		'num_divisions': 5,
		'd_use_hamiltonian': 1

	})
ham1_filt = Filtration(ellipse_traj, ham1_params, save=False)
