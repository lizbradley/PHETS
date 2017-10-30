import pytest
import sys, os

def pytest_sessionstart(session):
	print 'testing...'
	print os.path.realpath(__file__)
	sys.path.append(os.path.dirname(__file__))
