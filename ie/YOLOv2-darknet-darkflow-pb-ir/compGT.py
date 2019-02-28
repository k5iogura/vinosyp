#!/usr/bin/env python3
import math
import numpy as np
import sys,os
import argparse

def IntersectionOverUnion(box_1, box_2):
    xmin=0
    ymin=1
    xmax=2
    ymax=3
    width_of_overlap_area  = min(box_1[xmax], box_2[xmax]) - max(box_1[xmin], box_2[xmin])
    height_of_overlap_area = min(box_1[ymax], box_2[ymax]) - max(box_1[ymin], box_2[ymin]);
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1[ymax] - box_1[ymin])  * (box_1[xmax] - box_1[xmin])
    box_2_area = (box_2[ymax] - box_2[ymin])  * (box_2[xmax] - box_2[xmin])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0: return 0.0
    return area_of_overlap / area_of_union

def read_box(filename,w=1.0,h=1.0):
    f = open(filename)
    lines = f.readlines()
    ary = []
    for l in lines:
        l = l.strip().split(' ')
        ary.append(l)
    ary = np.asarray(ary,dtype=np.float32).reshape(-1)
    ary[1::5] = w*ary[1::5]
    ary[2::5] = h*ary[2::5]
    ary[3::5] = w*ary[3::5]
    ary[4::5] = h*ary[4::5]
    ary = ary.reshape(-1,5)
    return ary

def compare_with_IOU(GT,pr,iou_thresh=0.5,verbose=False):
    assert type(GT) == np.ndarray,'arg gt must be numpy.array'
    assert type(pr) == np.ndarray,'arg pr must be numpy.array'
    gt = GT.copy()
    if verbose:print("GroundTrush:%d Predicted:%d"%(gt.shape[0], pr.shape[0]))
    TP = []
    FP = []
    FN = []
    if verbose:print("# TP GT and iou")
    for pi in range(pr.shape[0]):
        predict_bbox = pr[pi][1:]
        matched=False
        for gi in range(gt.shape[0]):
            ground_bbox = gt[gi][1:]
            iou = IntersectionOverUnion(predict_bbox, ground_bbox)
            iou = int(10000.*iou)/10000.
            if iou > iou_thresh:
                if verbose:print(predict_bbox.astype(np.int),ground_bbox.astype(np.int),iou)
                TP.append(predict_bbox)
                matched=True
                gt[gi]=np.zeros((5),dtype=np.float32)
                break
        if not matched:
            FP.append(predict_bbox)
    TP = np.asarray(TP)
    FP = np.asarray(FP)
    FN = gt[gt>0.0].reshape(-1,5)
    return TP, FP, FN

def diff_location_region(pred, grnd, scale_er):
    xmin=1
    ymin=2
    xmax=3
    ymax=4
    assert type(pred) == np.ndarray,'arg pred must be numpy.array'
    assert type(grnd) == np.ndarray,'arg grnd must be numpy.array'
    diff_confidence = abs(pred[0] - grnd[0])
    diff_location = np.abs(pred[1:] - grnd[1:])
    w_1 = pred[xmax] - pred[xmin]
    h_1 = pred[ymax] - pred[ymin]
    w_2 = grnd[xmax] - grnd[xmin]
    h_2 = grnd[ymax] - grnd[ymin]
    thresh_reg  = max(w_2*scale_er, h_2*scale_er)
    diff_region = np.asarray([ w_1 - w_2, h_1 - h_2 ],dtype=np.float32)
    diff_region = np.abs(diff_region)
    return diff_confidence, diff_location, diff_region, thresh_reg

def compare_with_hike(GT, pr, errorSpec, verbose=False):
    assert type(GT) == np.ndarray,'arg gt must be numpy.array'
    assert type(pr) == np.ndarray,'arg pr must be numpy.array'
    gt = GT.copy()
    thresh_conf = errorSpec['conf']
    thresh_con  = errorSpec['ec']
    thresh_loc  = errorSpec['el']
    thresh_reg  = errorSpec['er']
    if not thresh_conf:thresh_con = 1000.0
    if verbose:print("GroundTrush:%d Predicted:%d"%(gt.shape[0], pr.shape[0]))
    TP = []
    FP = []
    FN = []
    if verbose:print("# DIFF BTN PR VS GT :LOCATION REGION and maxes")
    for pi in range(pr.shape[0]):
        predict_bbox = pr[pi][0:]
        matched=False
        for gi in range(gt.shape[0]):
            ground_bbox = gt[gi][0:]
            conf, loc, reg, reg_er = diff_location_region(predict_bbox,ground_bbox,thresh_reg)
            loc_max = np.max(loc)
            reg_max = np.max(reg)
    #        if verbose:print(loc.astype(np.int), reg.astype(np.int), loc_max, reg_max)
            if conf <= thresh_con and loc_max <= thresh_loc and reg_max <= reg_er:
                if verbose:print(predict_bbox.astype(np.int), ground_bbox.astype(np.int))
                TP.append(predict_bbox)
                matched=True
                gt[gi]=np.zeros((5),dtype=np.float32)
                break
        if not matched:
            FP.append(predict_bbox)
    TP = np.asarray(TP)
    FP = np.asarray(FP)
    FN = gt[gt>0.0].reshape(-1,5)
    return TP, FP, FN

def calc_precision_recall(TP,FP,FN):
    assert type(TP) == np.ndarray or type(TP) == list, "needs ndarray or list"
    assert type(FP) == np.ndarray or type(FP) == list, "needs ndarray or list"
    assert type(FN) == np.ndarray or type(FN) == list, "needs ndarray or list"
    nTP=len(TP)
    nFP=len(FP)
    nFN=len(FN)
    if (nTP+nFP)!=0:
        precision = nTP/(nTP+nFP)
    else:
        precision = 0.0
    if (nTP+nFN)!=0:
        recall    = nTP/(nTP+nFN)
    else:
        recall    = 0.0
    print(" # TP FP FN Precision Recall")
    print("   %d %d %d %.4f %.4f"%(nTP,nFP,nFN,precision,recall))
    print("")
    return precision, recall

if __name__ == '__main__':
    args = argparse.ArgumentParser('calcurate precision etc.')
    args.add_argument("-g", "--gt", type=str, nargs='+', help="Ground Truth files")
    args.add_argument("-p", "--pr", type=str, nargs='+', help="Prediction result files")
    args.add_argument("-i", "--iou",type=float, default=0.5, help="IOU Threshold")
    args.add_argument("-k", "--ke",     action='store_true', help="HIKE Criteria")
    args.add_argument("-c", "--conf",   action='store_true', help="HIKE Criteria with confidence")
    args.add_argument("-ec","--errC",type=float, default=0.01,help="HIKE Mode Error Confidence")
    args.add_argument("-el","--errL",type=float, default=2.0, help="HIKE Mode Error Location")
    args.add_argument("-er","--errR",type=float, default=0.05,help="HIKE Mode Error Region")
    args.add_argument("-x","--both_pixelscale",  action='store_true', help="infiles are both pixel scale")
    args.add_argument("-v", "--verbose",action='store_true', help="Verbose for debug")
    args = args.parse_args()
    assert len(args.gt)==len(args.pr), 'mismatched number of files'

    iou_thresh = args.iou
    thresh_conf= 0.01
    if args.ke:print("KE-Mode")

    TP=[]
    FP=[]
    FN=[]
    scale_w = 640
    scale_h = 480
    for gt_file,pr_file in zip(args.gt, args.pr):
        print(gt_file, pr_file)
        if args.both_pixelscale:scale_w = scale_h = 1.0
        gt_bbox=read_box(gt_file,w=1.0,h=1.0)
        pr_bbox=read_box(pr_file,w=scale_w,h=scale_h)

        #args.errC = args.errL = args.errR =  100.  # For Debugging
        if args.ke:
            errorSpec={
                'conf':args.conf,
                'ec':args.errC,
                'el':args.errL,
                'er':args.errR,
            }
            tp, fp, fn = compare_with_hike(
                gt_bbox,
                pr_bbox,
                errorSpec=errorSpec,
                verbose=args.verbose
            )
        else:
            tp, fp, fn = compare_with_IOU(gt_bbox,pr_bbox,iou_thresh,args.verbose)
        calc_precision_recall(tp,fp,fn)
        TP.extend(tp)
        FP.extend(fp)
        FN.extend(fn)
    TP=np.asarray(TP,dtype=np.float32)
    FP=np.asarray(FP,dtype=np.float32)
    FN=np.asarray(FN,dtype=np.float32)
    print('*'*28)
    print("** Total Precision/Recall **")
    print('*'*38)
    calc_precision_recall(TP,FP,FN)

