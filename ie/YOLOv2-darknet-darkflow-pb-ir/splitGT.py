#!/usr/bin/env python3
import sys,os
import numpy as np
import argparse
args = argparse.ArgumentParser('split Ground Truth etc.')
args.add_argument("files", nargs='+', help="GT or Prediction Data")
args.add_argument("-6", "--item6",action='store_true', help="specify 6 items")
args = args.parse_args()

for filename in args.files:
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
                if not args.item6:
                    line = "{} {} {} {} {}\n".format(i[0],i[1],i[2],i[3],i[4])
                else:
                    line = "{} {} {} {} {}\n".format(i[1],i[2],i[3],i[4],i[5])
                o.write(line)
