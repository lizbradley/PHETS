import numpy as np
from PRFCompare.Data import norm

dists_11 = np.loadtxt('PRFCompare/text_data/dist_1_vs_1.txt')		# top left
dists_12 = np.loadtxt('PRFCompare/text_data/dist_1_vs_2.txt')		# bottom left
dists_21 = np.loadtxt('PRFCompare/text_data/dist_2_vs_1.txt')		# top right
dists_22 = np.loadtxt('PRFCompare/text_data/dist_2_vs_2.txt')		# bottom right

variance_11 = np.mean(dists_11)
variance_12 = np.mean(dists_12)
variance_21 = np.mean(dists_21)
variance_22 = np.mean(dists_22)

print variance_11, variance_12, variance_21, variance_22			# checks out against plot


mean_prf_1 = np.loadtxt('PRFCompare/text_data/mean_1.txt')
mean_prf_2 = np.loadtxt('PRFCompare/text_data/mean_2.txt')

print mean_prf_1		# prints upside down :/

means_diff = np.subtract(mean_prf_1, mean_prf_2)
means_dist = norm(means_diff, 'L2', lambda i, j: 1)					# for that one statistic



