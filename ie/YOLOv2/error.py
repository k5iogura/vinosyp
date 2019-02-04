#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv)!=3:
    print("Usage: %s text1 text2"%sys.argv[0])
    sys.exit(-1)

with open(sys.argv[-2]) as inx:
    l1 = inx.read().strip().split()
print("%s len=%d"%(sys.argv[-2],len(l1)))

with open(sys.argv[-1]) as inx:
    l2 = inx.read().strip().split()
print("%s len=%d"%(sys.argv[-2],len(l2)))

l1 = np.asarray(l1,dtype=np.float64)
l2 = np.asarray(l2,dtype=np.float64)

#for i in zip(l1,l2):
    #[i0, i1]=(np.float32(i[0]), np.float32(i[1]))
    #if i0 != i1:
        #print(i0,i1)

(std_l1, std_l2 ) = np.std(l1), np.std(l2)
(max_l1, max_l2 ) = np.max(l1), np.max(l2)
(min_l1, min_l2 ) = np.min(l1), np.min(l2)

print("%44s"%"std/max/min")
print("%30s ="%(sys.argv[--2]),"%.10f %.10f %.10f"%(std_l1, max_l1, min_l1))
print("%30s ="%(sys.argv[--1]),"%.10f %.10f %.10f"%(std_l2, max_l2, min_l2))
