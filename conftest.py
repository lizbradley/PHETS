import pytest
import sys, os

def pytest_sessionstart(session):
	sys.path.append(os.path.dirname(__file__))
