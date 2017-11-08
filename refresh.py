import sys
import os
import subprocess
from utilities import remove_files_by_type
from config import find_landmarks_c_compile_str



# remove all .pyc and
for subdir, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pyc'):
            fname = os.path.join(subdir, file)
            os.remove(fname)
            print 'removed {}'.format(fname)



remove_files_by_type('DCE/frames/', '.txt')
remove_files_by_type('DCE/temp_data/', '.txt')


try:
    os.remove('PH/landmark_outputs.txt')
except OSError:
    pass

try:
    os.remove('PH/GI_edge_filtration.txt')
except OSError:
    pass

remove_files_by_type('PH/frames/', '.png')
remove_files_by_type('PH/perseus/', '.txt')



print 'recompiling find_landmarks.c...'
os.chdir('PH')
if sys.platform == "linux" or sys.platform == "linux2":
    compile_str = find_landmarks_c_compile_str['linux']
elif sys.platform == 'darwin':
    compile_str = find_landmarks_c_compile_str['macOS']
else:
    print 'Sorry, PHETS requires linux or macOS.'
    sys.exit()
subprocess.call(compile_str, shell=True)
os.chdir('..')
print 'recompilation attempt complete'

