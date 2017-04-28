from os import system, chdir
import sys
import os.path
import matplotlib.pyplot as plt

def MI(input_file, in_path = "/Users/nicolesanderson/Desktop/DCEer/input/piano_data/C135B", tisean_path = "/Users/nicolesanderson/Software/Tisean"):

	
	out = input_file[:-4] + '_MI'
	print 'The output file will be called %s.' % out
	in_file = in_path + '/' + input_file
	print 'The input file wil be found at %s.' % in_file 

	chdir(tisean_path)
	system('./mutual -D 100 -o %s %s' %  (out, in_file))
	
	out_file = tisean_path + '/' + out 
	
	chdir('/Users/nicolesanderson/Desktop/DCEer/input/piano_data/C135B')

	f = open(out_file, 'r')
	a = []
	j = 0
	for line in f:
		if "#" not in line:
			a.append(line.split()[1])
		
		
	fig=plt.figure(figsize=(12,10))
	plt.plot(a)
	plt.title("Delay vs Mutual Information: %s" % input_file)
	#plt.show()
	fig.savefig("%s.png" % out )
	
MI(sys.argv[1], in_path = "/Users/nicolesanderson/Desktop/DCEer/input/piano_data/C135B", tisean_path = "/Users/nicolesanderson/Software/Tisean")