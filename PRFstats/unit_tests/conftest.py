# import os
# import pytest
# import sys
# from paths import current_dir
#
# def pytest_sessionstart(session):
# 	print 'hiii'
# 	print current_dir
# 	os.chdir(current_dir)
#
#
# def pytest_sessionfinish(session):
# 	print 'remove cd', current_dir
# 	sys.path.remove(current_dir)
