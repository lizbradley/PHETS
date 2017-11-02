import os
import sys
import time


def change_dir():
	root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
	sys.path.append(root_path)
	os.chdir(root_path)


def get_test(set_test):

	if len(sys.argv) > 1:
		test = int(sys.argv[1])
	else:
		test = set_test
	print 'running test %d...' % test
	start_time = time.time()

	return test, start_time
