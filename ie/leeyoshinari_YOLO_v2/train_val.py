#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:leeyoshinari
#-----------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import argparse
import datetime
import time
import os,sys
import yolo.config as cfg

from pascal_voc import Pascal_voc
from six.moves import xrange
from yolo.yolo_v2 import yolo_v2
import random
# from tensorflow.python import debug as tf_debug # for tfdebugger
# from yolo.darknet19 import Darknet19

from pdb import *
class Train(object):
    def __init__(self, yolo, data, optimizer_no=1, var_set='all'):
        self.yolo = yolo
        self.data = data
        self.num_class = len(cfg.CLASSES)
        self.max_step = cfg.MAX_ITER
        self.saver_iter = cfg.SAVER_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.initial_learn_rate = cfg.LEARN_RATE
        self.output_dir = os.path.join(cfg.DATA_DIR, 'output')
        weight_file = os.path.join(self.output_dir, cfg.WEIGHTS_FILE)

        self.variable_to_restore = tf.global_variables()
#        for i in self.variable_to_restore:
#            print(i.name)      # Print out variable names
        #self.saver = tf.train.Saver(self.variable_to_restore)  # for restore full
        print("** make saver with restored/all_variables=%d/%d"%(len(yolo.frontV),len(self.variable_to_restore)))
        print("** make saver with all variables",len(self.variable_to_restore))
        self.loss_min_train = self.loss_min_test = 1e100
        self.saver_front = tf.train.Saver(yolo.frontV)                # for restore part of yolo_v2()
        self.saver_back  = tf.train.Saver(yolo.backV)                # for restore part of yolo_v2()
        self.saver_full  = tf.train.Saver(self.variable_to_restore)

        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate')
        # self.global_step = tf.Variable(0, trainable = False)
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 190, 10000, 15500], [1e-3, 5e-3, 1e-2, 1e-3, 1e-4])
        print("** var_set for opt:", var_set)
        if var_set == 'all':
            var4opt = None          # Optimize all variables
        else:
            var4opt = yolo.backV    # Optimize a part of variable in Graph

        if   optimizer_no == 1:
            self.optimizer=tf.train.AdagradOptimizer(
                learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step, var_list=var4opt
            )
        elif optimizer_no == 2:
            self.optimizer=tf.train.AdamOptimizer(
                learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step, var_list=var4opt
            )
        elif optimizer_no == 3:
            self.optimizer=tf.train.GradientDescentOptimizer(
                learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step, var_list=var4opt
            )
        else:
            self.optimizer=tf.train.AdagradDAOptimizer(
                learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step, var_list=var4opt
            )
        print("** Selected Optimizer:",optimizer_no,self.optimizer.name)
        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.average_op)

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            #allow_growth=True,
            #per_process_gpu_memory_fraction=0.3    # cause segv
        ))
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer(),options=run_options)

        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)    # tfdebugger

        print('** Restore weights from:', weight_file)
        self.saver_front.restore(self.sess, weight_file)
        self.writer.add_graph(self.sess.graph)

    def train(self):
        labels_train = self.data.load_labels('train')
        labels_test = self.data.load_labels('test')

        num = 5
        loss = loss_t = 1e100
        initial_time = time.time()

        for step in xrange(0, self.max_step + 1):
            images, labels = self.data.next_batches(labels_train)
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels}

            if step % self.summary_iter == 0:
                if step % 50 == 0:
                    summary_, loss, _ = self.sess.run([self.summary_op, self.yolo.total_loss, self.train_op], feed_dict = feed_dict)
                    sum_loss = 0

                    for i in range(num):
                        images_t, labels_t = self.data.next_batches_test(labels_test)
                        feed_dict_t = {self.yolo.images: images_t, self.yolo.labels: labels_t}
                        loss_t = self.sess.run(self.yolo.total_loss, feed_dict=feed_dict_t)
                        sum_loss += loss_t
                    loss_t = sum_loss/num

                    log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f}, test_Loss: {:.4f}, Remain: {}').format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss, loss_t, self.remain(step, initial_time))
                    sys.stdout.write(log_str)

                    if loss != loss:	# loss is NaN
                        print('** loss is NaN')
                        break
                    elif loss > 1e4:
                        print('** loss > 1e4')
                        break

                else:
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)

                self.writer.add_summary(summary_, step)

            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)

            if step % self.saver_iter == 0 and loss == loss and loss_t == loss_t:
                if self.loss_min_train > loss or self.loss_min_test > loss_t:
                    self.saver_full.save(self.sess, self.output_dir + '/yolo_v2_FTune.ckpt', global_step = step)
                    sys.stdout.write(', saved')
                if self.loss_min_train > loss:   self.loss_min_train = loss
                if self.loss_min_test  > loss_t: self.loss_min_test  = loss_t
            sys.stdout.write('\n')

    def remain(self, i, start):
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--weights',  default=None, type = str)  # darknet-19.ckpt
    parser.add_argument('--weight_dir',    default = 'output', type = str)
    parser.add_argument('--data_dir',      default = 'data',   type = str)
    parser.add_argument('-o','--optimizer',default = 1,        type = int)
    parser.add_argument('-v','--var_set',  default = 'all',    type = str, choices=['all','back'])
    parser.add_argument('-g','--gpu',      default = '',       type = str)  # which gpu to be selected
    args = parser.parse_args()

    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    tf.set_random_seed(cfg.RANDOM_SEED)

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.weights is not None:
        cfg.WEIGHTS_FILE = args.weights
    else:
        w_dir  = (os.path.join(cfg.DATA_DIR,args.data_dir))
        latest = tf.train.latest_checkpoint(w_dir)
        if latest is not None and len(latest)>0: cfg.WEIGHTS_FILE = latest
    print("** resore weights file:",cfg.WEIGHTS_FILE)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    yolo = yolo_v2()
    # yolo = Darknet19()
    pre_data = Pascal_voc()

    train = Train(yolo, pre_data, optimizer_no=args.optimizer, var_set=args.var_set)

    print('** start training ...')
    train.train()
    print('** successful training.')


if __name__ == '__main__':
    main()
