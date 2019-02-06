#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv)<=1:
    print("Usage: %s text1 text2 ..."%sys.argv[0])
    sys.exit(-1)

def readtxt(filename):
    with open(filename) as inx:
        l = inx.read().strip().split()
    lines = len(l)
    l = np.asarray(l,dtype=np.float32)
    std_l = np.std(l)
    max_l = np.max(l)
    min_l = np.min(l)
    return (lines, std_l, max_l, min_l)

print("%44s"%"std/max/min")

for f in sys.argv[1:]:
    lines, std_l, max_l, min_l = readtxt(f)
    print("%30s ="%(f),"%.8f %.8f %.8f"%(std_l, max_l, min_l))
