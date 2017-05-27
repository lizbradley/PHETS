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