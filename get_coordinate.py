import numpy as np

file = np.loadtxt('datasets/trajectories/btc2milIC123_embedded.txt')

file = file[:,1]

np.savetxt('datasets/trajectories/btc2milIC123_coordinate1.txt',file)

