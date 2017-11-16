from signals import TimeSeries, Trajectory

clar_ts = TimeSeries('data/40-clarinet.txt', crop=(1000, 10000),
                     num_windows=15, vol_norm=(1, 1, 1))


ellipse_traj = Trajectory('data/ellipse.txt', crop=(100, 900), num_windows=15,
                          vol_norm=(1, 1, 1))


