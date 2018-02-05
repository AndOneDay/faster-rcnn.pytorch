# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------


import pickle
import os

import datetime
from typing import List

import numpy as np
# Allow matplotlib to run without $DISPLAY
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt

from .imdb import imdb


class DIT(imdb):
    def __init__(self, image_set, root_path, devkit_path, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: train, val, test
        :param devkit_path: data and results
        :return: imdb object
        """

        self.data_path = devkit_path
        self.image_set = image_set
        self.root_path = root_path
        self.devkit_path = devkit_path
        self.result_path = result_path

        super(DIT, self).__init__('dit')

        self._classes = ('__background__',  # always index 0
                        'person')

        self._image_index = self.load_image_set_index()
        self._roidb_handler = self.gt_roidb()
        self._time = datetime.datetime.now()

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'imageSets',
                                            self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(
            image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'images', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_'+ self.image_set + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'annotations', index + '.xml')
        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)

        objs = tree.findall('object')

        ignore = [x for x in objs if x.find('name').text.lower() == 'ignore']
        objs = [x for x in objs if x.find('name').text.lower() in self.classes]

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.uint16)

        class_to_index = dict(list(zip(self.classes, list(range(self.num_classes)))))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

            # GT is difficult either if annotator said so (>0.5 occlusion) or if it's too small.
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            if x2 - x1 < 25 or y2 - y1 < 50:
                difficult = 1
            ishards[ix] = difficult

        # Load ignore/don't care
        dont_care = np.zeros((len(ignore), 4), dtype=np.uint16)
        for ix, obj in enumerate(ignore):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            dont_care[ix, :] = [x1, y1, x2, y2]

        roi_rec.update({'boxes':        boxes,
                        'gt_classes':   gt_classes,
                        'gt_overlaps':  overlaps,
                        'gt_ishard':    ishards,
                        'max_classes':  overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped':      False,
                        'dont_care':    dont_care})
        return roi_rec

    def evaluate_detections(self, detections, easy=False, **kwargs):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :param easy: whether to use all dets or just easy ones during evalution
        :param cfg_name: name of configuration used
        :return: None
        """

        self.write_pascal_results(detections)
        self.do_python_eval(easy=easy)

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        res_file_folder = os.path.join(self.result_path, 'results')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        filename = 'det_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} {} results file'.format(cls, self.name))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1,
                                       dets[k, 3] + 1))

    def do_python_eval(self, easy=False):
        # Put other configuration names in this list
        filename = self.get_result_file_template()
        results = self.plot_multiple_results([filename], easy=easy,
                                             save_name=self._time.strftime("%m-%d %H:%M"))
        print('--------------------------------------------------------------')
        print(('{:>25s}: {:^5s} {:^5s} ({:^5s})'.format('Name', 'AP', 'MR2', 'MR4')))
        for name, (ap, mr2, mr4) in list(results.items()):
            print(('{:>25s}: {:4.3f} {:4.3f} ({:4.3f})'.format(name, ap, mr2, mr4)))
        print('--------------------------------------------------------------')

    def eval(self, detpath, class_idx, ovthresh=0.5, easy=False):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, self.image_set + '.pkl')
        # read list of images

        if not os.path.exists(cachefile):
            recs = self.gt_roidb()
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            print('Loading cached annotation from {:s}'.format(cachefile))
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f, encoding='latin1')
        # extract gt objects for this class
        # govind: recs is not class specific. Hence create another
        # dictionary class_recs which is specific to this class
        class_recs = {}
        npos = 0
        nimg = len(recs)
        for i, img in enumerate(recs):
            R = [obj for idx, obj in enumerate(img['boxes']) if img['gt_classes'][idx] ==
                 class_idx]
            hard = [img['gt_ishard'][idx] for idx, _ in enumerate(img['boxes']) if
                    img['gt_classes'][idx] == class_idx]
            bbox = np.array(R)
            det = [False] * len(R)
            npos += len(R) - (np.sum(hard) if easy else 0)
            class_recs[i] = {'bbox': bbox, 'det': det, 'hard': hard}

        detfile = detpath.format(self.classes[class_idx])
        with open(detfile, 'r') as f:
            lines = f.readlines()

        splitlines = [x.strip().split(' ') for x in lines]
        # Sort by confidence
        splitlines = sorted(splitlines, key=lambda k: float(k[1]), reverse=True)
        # go down dets and mark TPs and FPs
        nd = len(splitlines)

        TP = 0.
        FP = 0.
        MR = np.ones(nd)
        FPPI = np.zeros(nd)
        RECALL = np.zeros(nd)
        PRECISION = np.zeros(nd)
        missed = set()
        for idx, line in enumerate(splitlines):
            img_idx = self.image_index.index(line[0])
            if img_idx not in class_recs:
                missed.add(line[0])
                MR[idx] = 1. - TP / npos
                FPPI[idx] = FP / nimg
                RECALL[idx] = TP / npos
                PRECISION[idx] = TP / (TP + FP) if TP or FP else 1
                continue
            R = class_recs[img_idx]
            bb = np.array(line[2:]).astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            MR[idx] = 1. - TP / npos
            FPPI[idx] = FP / nimg
            RECALL[idx] = TP / npos
            PRECISION[idx] = TP / (TP + FP) if TP or FP else 1

            if ovmax > ovthresh:
                if easy and R['hard'][jmax]:
                    R['det'][jmax] = 1
                    continue
                if not R['det'][jmax]:
                    R['det'][jmax] = 1
                    TP += 1.
                else:
                    FP += 1.
            else:
                FP += 1.

        if len(missed) > 10:
            print(("Missed:", len(missed)))
        elif missed:
            print(("Missed:", missed))

        # correct AP calculation
        # first append sentinel values at the enad
        mrec = np.copy(RECALL)
        mpre = np.copy(PRECISION)

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        AP = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return RECALL, PRECISION, AP, MR, FPPI

    def get_results_dir(self, output_dir='results'):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        return os.path.join(self.data_path, output_dir)

    def single_file_eval(self, filename, easy=False):
        class_points = {}
        class_results = {}
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # govind: It's calling a function which will give Precision, Recall and
            # average precision if we pass it the _results_file
            rec, prec, ap, mr, fppi = self.eval(filename, i, easy=easy)
            class_points[cls] = rec, prec, mr, fppi

            # Caltech dataset reports performance in log-average miss rate on results with mr
            #  between 1e-2 and 0.
            mr4i = next(idx for idx, element in enumerate(fppi) if element > 1e-4)
            mr2i = next(idx for idx, element in enumerate(fppi) if element > 1e-2)
            try:
                mr0i = next(idx for idx, element in enumerate(fppi) if element > 1.)
            except StopIteration:
                mr0i = -1
            mr4 = mr[mr4i:mr0i]
            mr2 = mr[mr2i:mr0i]
            log_average_mr2 = np.exp(np.mean(np.log(mr2)))
            log_average_mr4 = np.exp(np.mean(np.log(mr4)))
            class_results[cls] = ap, log_average_mr2, log_average_mr4
        return class_results, class_points

    def plot_multiple_results(self, detection_files: List[str], easy=False, save_name: str = None):
        """
        Creates and saves a graph of results
        :param detection_files: list of detection filenames
        :param easy: whether to use only easy detections
        :param save_name: filename to use for graph
        """
        if save_name is None:
            save_name = ':'.join([x.split('/')[1]
                                  for x in detection_files]) + self._time.strftime(" %m-%d %H:%M")

        aps = {}
        plt.figure(figsize=(24, 9))
        fig = plt.gcf()
        fig.suptitle("DIT:{} results ({})".format(self.image_set, "Easy" if easy else 'All'))

        for filename in detection_files:
            class_results, class_points = self.single_file_eval(filename)
            for i, cls in enumerate(self.classes):
                # Exclude background
                if cls == '__background__':
                    continue
                rec, prec, mr, fppi = class_points[cls]
                ap, log_average_mr2, log_average_mr4 = class_results[cls]
                name = "{} ({})".format(cls, filename.split('/')[1])
                np.set_printoptions(precision=3)
                # Miss rate vs FPPI
                plt.subplot(1, 2, 1)
                plt.scatter(fppi, mr, s=.2, label='{:>25s}: {:3.3f} ({:3.3f})'.format(
                    name, log_average_mr2, log_average_mr4))
                # PR vs RC
                plt.subplot(1, 2, 2)
                plt.scatter(rec, prec, s=.2, label='{:>25s}: {:3.3f}'.format(name, ap))
                aps[name] = ap, log_average_mr2, log_average_mr4

        plt.subplot(1, 2, 1)
        plt.ylabel("Miss rate")
        plt.xlabel("FPPI")
        plt.xlim([1e-3, 10.])
        plt.ylim([1e-2, 1.05])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both')
        ax = plt.gca()
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        plt.legend(prop={'family': 'monospace'})
        # PR vs RC
        plt.subplot(1, 2, 2)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.xlim([-.05, 1.05])
        plt.ylim([-.05, 1.05])
        plt.grid(True)
        plt.legend(prop={'family': 'monospace'})

        resultsdir = self.get_results_dir()
        os.makedirs(resultsdir, exist_ok=True)
        plt.savefig(os.path.join(resultsdir, save_name + ".png"),
                    orientation="landscape", bbox_inches='tight')
        return aps
