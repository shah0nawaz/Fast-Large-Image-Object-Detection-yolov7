import os
import glob


with open('list.txt', 'w') as f:
    for fil in glob.glob('./inference/images/*.jpeg'):
        f.write(fil + '\n')
