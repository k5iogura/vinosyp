import sys,os
import fnmatch
import numpy as np
directory="VOCdevkit/VOC2007/"

files=[]
for root,dirs,names in os.walk(directory):                     
    finded_files=[os.path.join(root,f) for f in names      
        if fnmatch.fnmatch(os.path.join(root,f), '*labels/*.txt')] 
    files.extend(finded_files) 

jpeg_dir=os.path.join(os.environ["PWD"],"VOCdevkit/VOC2007/JPEGImages/")
assert os.path.exists(jpeg_dir)

print("filename,rois,classes")
for f in files:
    with open(f) as fp:
        lines = np.asarray(fp.read().strip().split(),dtype=np.float32).reshape(-1,5)
    labels = np.asarray([lines[i][0] for i in range(len(lines))], dtype=np.int)
    coords = np.asarray([lines[i][1:] for i in range(len(lines))], dtype=np.float32)

    f = os.path.join(jpeg_dir,os.path.splitext(os.path.basename(f))[0])+'.jpg'
    assert os.path.exists(f)

    sys.stdout.write(f+",\"[")

    for no,i in enumerate(coords):
        sys.stdout.write("[%.3f,%.3f,%.3f,%.3f]"%(i[0],i[1],i[2],i[3]))
        if no != len(coords)-1:
            sys.stdout.write(',')
        if no == len(coords)-1:
            sys.stdout.write("]\"")

    sys.stdout.write(",\"[")
    for no,i in enumerate(labels):
        sys.stdout.write("%d"%(i))
        if no != len(coords)-1:
            sys.stdout.write(',')
        if no == len(coords)-1:
            sys.stdout.write("]\"")
    sys.stdout.write('\n')
