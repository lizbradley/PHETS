import subprocess

def test__dce():
	subprocess.call(['python', 'dceTester.py', '-1'])

def test__ph():
	subprocess.call(['python', 'phTester.py', '-1'])

def test__prfsTester():
	subprocess.call(['python', 'prfsTester.py', '-1'])

def test__demo():
	subprocess.call(['python', 'demoTester.py', '-1'])
