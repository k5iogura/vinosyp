#!/usr/bin/env python3
import sys,os
import numpy as np
for filename in sys.argv[1:]:
    f = open(filename)
    lines = f.readlines()
    f.close()
    for l in lines:
        l = l.strip().split(',')
        imgfile = str(l[0])
        objects = int(l[1])
        coords  = np.asarray(l[2:],dtype=np.float32).reshape((objects,-1))
        newname = os.path.splitext(imgfile)[0]+'.gt'
        coords[:,0,] = 1.0  # confidence of person is always 100%
        #print(coords)
        with open(newname,'w') as o:
            print('Created',newname, 'objects', objects)
            for i in coords:
                line = "{} {} {} {} {}\n".format(i[0],i[1],i[2],i[3],i[4])
                #o.write("{} {} {} {} {}\n".format(i[0],i[1],i[2],i[3],i[4]))
                o.write(line)
