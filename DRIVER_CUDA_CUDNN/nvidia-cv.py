#! /usr/bin/env python
import subprocess
import argparse
import cv2
import sys
import re
import numpy as np
from pdb import *

def res_cmd(cmd):
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        shell=True
    ).stdout.readlines()

def nvidia_smi():
    gpu_line=0
    fan=[]
    temp=[]
    power=[]
    usedmem=[]
    maxmem=[]
    usedgpu=[]
    for l in res_cmd('nvidia-smi'):
        lstr=str(l.strip())
        if re.findall('iB',lstr) and re.findall('C',lstr) and re.findall('W',lstr):
            word=lstr.split()
            ifan             = re.findall('[N/A0-9]+',word[1])[0]
            itemp            = int(re.findall('[0-9]+',word[2])[0])
            ipower           = int(re.findall('[0-9]+',word[4])[0])
            iusedmem         = int(re.findall('[0-9]+',word[8])[0])
            imaxmem          = int(re.findall('[0-9]+',word[10])[0])
            iusedgpu         = int(re.findall('[0-9]+',word[12])[0])
            if ifan=='N/A': ifan = 0
            fan.append(int(ifan))
            temp.append(itemp)
            power.append(ipower)
            usedmem.append(iusedmem)
            maxmem.append(imaxmem)
            usedgpu.append(iusedgpu)
            gpu_line+=1
        else:
            continue
    return gpu_line,fan,usedmem,maxmem,usedgpu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predictor for det')
    parser.add_argument('--gpu',  '-g', type=int, default=0)
    parser.add_argument('--width','-w', type=int, default=300)
    args = parser.parse_args()

    gpuN,_,_,_,_ = nvidia_smi()
    if gpuN==0:
        print('No GPU found')
        sys.exit(1)
    height = 100
    width  = args.width
    fig = np.zeros((gpuN*height+20,width,3),dtype=np.float32)
    #start_gpu = [(5,5),(5,100+5)]
    #start_mem = [(5,5),(5,100+5)]
    start_gpu = []
    start_mem = []
    for igpu in range(0,gpuN):
        start_gpu.append((5,igpu*100+1))
        start_mem.append((5,igpu*100+1))
    ix = 2
    while True:
        gpuN,_,usedmem,maxmem,usedgpu = nvidia_smi()
        for igpu in range(0,gpuN):
            oset = igpu*height
            #axis
            cv2.line(fig,(5,oset+height-5),  (5,oset+height-95),       (255,255,255),1)
            cv2.line(fig,(5,oset+height/2-5),(width-5,oset+height/2-5),(20,0,0),     1)
            cv2.line(fig,(5,oset+height-5),  (width-5,oset+height-5),  (255,255,255),1)
            Usedmem = int(height * float(usedmem[igpu])/maxmem[igpu])
            Usedgpu = usedgpu[igpu]
            #data
            cv2.line(fig,start_gpu[igpu],(ix,oset+height-Usedgpu),(0,0,255),1)
            cv2.line(fig,start_mem[igpu],(ix,oset+height-Usedmem),(0,255,0),1)
            start_gpu[igpu] = (ix,oset+height-Usedgpu)
            start_mem[igpu] = (ix,oset+height-Usedmem)
        ix+=2
        cv2.imshow('GPU-Status',fig)
        if ix > width-10:
            #start_gpu = [(5,5),(5,100+5)]
            #start_mem = [(5,5),(5,100+5)]
            start_gpu = []
            start_mem = []
            for igpu in range(0,gpuN):
                start_gpu.append((5,igpu*100+1))
                start_mem.append((5,igpu*100+1))
            fig = np.zeros((gpuN*height,width,3),dtype=np.float32)
            ix=5
        k = cv2.waitKey(500)
        if k==27:break
        if k==1048603:break
        if k==1310819:break
        elif k==-1:pass
        else:print('key=%d'%k)

    cv2.destroyAllWindows()
