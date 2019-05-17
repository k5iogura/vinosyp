#! /usr/bin/env python3
import sys,os,argparse
from pdb import *
import mmap
import numpy as np
import numpy

a = np.asarray([1,2,3,4],dtype=np.uint8)
a.tostring()

class devmem():
    def __init__(self, target_adr, length, verbose=False):
        self.verbose   = verbose
        self.target_adr= target_adr
        self.length    = length
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        self.reg_base  = int(target_adr // self.page_size) * self.page_size
        self.seek_size = int(target_adr %  self.page_size)
        if self.verbose:print("base adr:%s seek:%s"%(hex(self.reg_base), hex(self.seek_size)))

        self.fd  = os.open("/dev/mem", os.O_RDWR|os.O_SYNC)
        self.mem = mmap.mmap(
            self.fd,
            self.length+self.seek_size,
            mmap.MAP_SHARED,
            mmap.PROT_READ|mmap.PROT_WRITE,
            offset=self.reg_base
        )
        self.mem.seek(self.seek_size, os.SEEK_SET)

    def write(self, datas):
        self.mem.write(datas)
        self.mem.flush(0,0)
        return self

    def type_bytes(self, type):
        type_B = len(np.asarray(0,dtype=type).tostring())
        return type_B

    def read(self, types):
        type_bytes = self.type_bytes(types)
        assert self.length%type_bytes==0, "length {} may cause system freeze".format(self.length)
        ret = []
        for i in range(0, self.length, type_bytes):
            datas = self.mem.read(type_bytes)
            array = np.fromstring(datas,dtype=types)
            if types is np.uint32:
                if self.verbose:print("Value at address %s : 0x%8x"%(hex(self.target_adr+i),array[0]))
            elif types is np.uint8:
                if self.verbose:print("Value at address %s : 0x%2x"%(hex(self.target_adr+i),array[0]))
            else:
                if self.verbose:print("Value at address {} : {}".format(hex(self.target_adr+i),array))
            ret.extend(array)
        return np.asarray(ret)

    def close(self):
        self.mem.close()

if __name__=='__main__':
    #def s2i(s):return int(s,16)
    def s2i(s): return eval(s)
    args = argparse.ArgumentParser('devmem')
    args.add_argument("target_adr",     type=s2i, default=0xe018c000,    help="0xe018c000 for Image")
    args.add_argument("-s", "--size",   type=s2i, default=4,             help="bytes default 4")
    args.add_argument("-t", "--type",   type=str, default="np.float32",  help="type default numpy.float32")
    args.add_argument("-f", "--weights",type=str, default="yolo.weights",help="weights default yolo.weights")
    args.add_argument("-w", "--write",  action='store_true',             help="write default read")
    args.add_argument("-S", "--silence",action='store_true',             help="silence")
    args = args.parse_args()

    verbose=True
    if args.silence: verbose=False
    if args.write:
        str = "d = np.arange(1,{}).astype({})".format(args.size,args.type)
        exec(str)
        d = d.tostring()
        print("write Bytes",len(d))
        devmem(args.target_adr,len(d),verbose=verbose).write(d).close()
    else:
        print("read Bytes",args.size)
        str = "devmem(0x{:08x},{},verbose=verbose).read({})".format(args.target_adr, args.size, args.type)
        print(str)
        exec(str)

