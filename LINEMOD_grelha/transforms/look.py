import numpy as np
import glob

l = glob.glob('*.npy')
l.sort() 
for f in l:
    t = np.load(f)
    print(f)
    print(t)