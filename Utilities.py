import os
import sys
from memory_profiler import profile


def mem_profile(f, flag):
	if flag: return profile(stream=f)

	else: return lambda x: x




def check_overwrite(out_file_name):
	os.chdir('output')
	if os.path.exists(out_file_name):
		overwrite = raw_input(out_file_name + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			print 'goodbye'
			sys.exit()
	os.chdir('..')


def clear_old_files(path, see_samples):
	old_files = os.listdir(path)
	if old_files and see_samples:
		ans = raw_input('Clear files in ' + path + '? (y/n/q) \n')
		if ans == 'y':
			for f in old_files:
				if f != '.gitkeep':
					os.remove(path + f)
		elif ans == 'q':
			print 'goodbye'
			sys.exit()
		else:
			print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'


def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__

