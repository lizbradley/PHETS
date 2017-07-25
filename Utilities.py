import os, sys, inspect
from memory_profiler import profile


def mem_profile(f, flag):
	if flag: return profile(stream=f)

	else: return lambda x: x



# consolidate following three functions

def check_overwrite(out_file_name):
	os.chdir('output')
	if os.path.exists(out_file_name):
		overwrite = raw_input(out_file_name + " already exists. Overwrite? (y/n)\n")
		if overwrite == "y":
			pass
		else:
			print 'Goodbye'
			sys.exit()
	os.chdir('..')


def clear_old_files(path, see_samples):
	old_files = os.listdir(path)
	if old_files and see_samples:
		ans = raw_input('Overwrite files in ' + path + '? (y/n/q) \n')
		if ans == 'y':
			for f in old_files:
				if f != '.gitkeep':
					os.remove(path + f)
		elif ans == 'q':
			print 'Goodbye'
			sys.exit()
		else:
			print 'Proceeding... conflicting files will be overwritten, otherwise old files will remain. \n'


def clear_dir(dir):

	files = os.listdir(dir)
	if files:
		r = raw_input('Overwrite files in {}? (y/n/q) '.format(dir))
	else:
		return True

	if r == 'y':
		for f in files:
			os.remove(dir + f)
		return True

	elif r == 'n':
		return False

	else:
		print 'Goodbye'
		sys.exit()


def lambda_to_str(f):
	return inspect.getsourcelines(f)[0][0].split(':')[1]

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__




def print_title(str):
	if str:
		print '=================================================================='
		print str
		print '=================================================================='


def count_lines(dir, blanks=True):

	def file_len(fname):
		count = 0
		with open(fname) as f:
			for line in f:
				if line != '\n' or blanks:
					count += 1
		return count

	count_f = 0
	count_l = 0
	for root, dirs, files in os.walk(dir, topdown=False):
		for name in files:
			if name.endswith('.py'):
				fname = os.path.join(root, name)
				length = file_len(fname)

				count_f += 1
				count_l += length

				print '{}: {}'.format(fname, length)

	print_title('num files: {}\tnum lines: {}'.format(count_f, count_l))

