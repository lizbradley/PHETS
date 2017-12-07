import os

root_dir = os.path.realpath(__file__).split('/PHETS')[0] + '/PHETS'
current_dir = os.path.realpath(os.path.dirname(__file__))

def chdir():
	os.chdir(current_dir)

