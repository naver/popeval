#-*- coding: utf-8 -*-
'''
Copyright (c) 2019-present NAVER Corp.



Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys
import glob
import os
import io
import shutil
import json
import os
import argparse
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.strtree import STRtree
import traceback

py2 = (sys.version_info < (3, 0))
if not py2:
    from functools import reduce

SPLIT_DELIMITER = '##::'

# reorder vertices 
def getPolygon(vertex_list): 
    polygon = MultiPoint(vertex_list).convex_hull
    return polygon

def chunker(seq, size):
    seq = [int(e) for e in seq ]
    chunked_list = [seq[pos:pos + size] for pos in range(0, len(seq), size)]
    return chunked_list

def _divide(a,b):
    return 0 if b==0 else a/b

def removeDoncareBox(gt_boxes, gt_texts, pred_boxes, pred_texts, doncare_text=None):
    if doncare_text == None:
        return gt_boxes, gt_texts, pred_boxes, pred_texts 
    else:
        pred_idx_removed_for_doncare_list = []
        gt_idx_removed_for_doncare_list = []
        for gt_idx, (gt_box, gt_text) in enumerate(zip(gt_boxes,gt_texts)):
            if gt_text == doncare_text:
                gt_idx_removed_for_doncare_list.append(gt_idx)
                for pred_idx, pred_box in enumerate(pred_boxes):
                    intersectionarea = gt_box.intersection(pred_box).area
                    if intersectionarea > 0:
                        darea = pred_box.area
                        gtarea = gt_box.area
                        iou = _divide(float(intersectionarea), float(darea) + float(gtarea) - float(intersectionarea))
                        area_precision = _divide(float(intersectionarea), darea)
                        if area_precision > 0.5:
                        #if iou > 0.5:
                            pred_idx_removed_for_doncare_list.append(pred_idx)
        new_pred_boxes = [x for new_pred_idx, x in enumerate(pred_boxes) if new_pred_idx not in pred_idx_removed_for_doncare_list]
        new_pred_texts = [x for new_pred_idx, x in enumerate(pred_texts) if new_pred_idx not in pred_idx_removed_for_doncare_list]
        new_gt_boxes = [x for new_gt_idx, x in enumerate(gt_boxes) if new_gt_idx not in gt_idx_removed_for_doncare_list]
        new_gt_texts = [x for new_gt_idx, x in enumerate(gt_texts) if new_gt_idx not in gt_idx_removed_for_doncare_list]

        return new_gt_boxes, new_gt_texts, new_pred_boxes, new_pred_texts

def removeNoncontroversialBox(gt_boxes, gt_texts, pred_boxes, pred_texts):
    #score initialized
    #copy list
    new_pred_texts = list(pred_texts)
    new_gt_texts = list(gt_texts)
    new_gt_boxes = list(gt_boxes)

    gt_box_del_candidates = {}
    for idx in range(len(gt_boxes)):
        gt_box_del_candidates[idx] = []


   
    # shapely polygon query optimization 
    gt_idx_by_id = dict((id(gt_box), i) for i, gt_box in enumerate(gt_boxes))
    gt_tree = STRtree(gt_boxes)
    for pred_idx, pred_box in enumerate(pred_boxes):
        searched_boxes = gt_tree.query(pred_box)
        searched_indices = [gt_idx_by_id[id(gt_box)] for gt_box in searched_boxes]
        searched_texts = [new_gt_texts[gt_idx] for gt_idx in searched_indices]
        for gt_text, gt_idx in zip(searched_texts, searched_indices):
            pred_text_set = set(new_pred_texts[pred_idx])
            for gt_char in gt_text:
                if gt_char in pred_text_set: # if any character in gt exists in prediction text, we put this to gt_box_del_candidates
                    gt_box_del_candidates[gt_idx].append(pred_idx)
                    break;

     # for gt_idx, gt_box in enumerate(gt_boxes):
        # for pred_idx, pred_box in enumerate(pred_boxes):
            # if gt_box.intersects(pred_box): 
                # gt_text = new_gt_texts[gt_idx]
                # for gt_char in gt_text:
                    # if gt_char in new_pred_texts[pred_idx]: # if any character in gt exists in prediction text,
                                                            # we put this to gt_box_del_candidates
                        # gt_box_del_candidates[gt_idx].append(pred_idx)
                        # break;

    # filtering non-controversial ones among gt_box_del_candidates 
    gt_idx_to_pred_idx_non_conv = {}
    for k in list(gt_box_del_candidates.keys()):
        if len(gt_box_del_candidates[k]) == 1:
            gt_idx_to_pred_idx_non_conv[k] = gt_box_del_candidates[k][0]

    # IF there is no gt_box which is non-controversial, EXIT THE RECURSIVE and return the results.
    if len((gt_idx_to_pred_idx_non_conv.keys())) == 0:
        return gt_boxes, new_gt_texts, pred_boxes, new_pred_texts, gt_box_del_candidates

    ###############################################################################
    ##### gt_box removal(matching with predictions) priority
    #####
    ##### distance between inner centroid and (0,0) point.(smaller is removed first)
    ##############################################################################
    keys = list(gt_idx_to_pred_idx_non_conv.keys())
    gt_box_origin_dists = []
    for gt_idx in keys:
        gt_box = gt_boxes[gt_idx]
        gt_box_origin_dist = Point(0,0).distance(gt_box.representative_point())
        gt_box_origin_dists.append(gt_box_origin_dist)
    gt_idx = keys[gt_box_origin_dists.index(min(gt_box_origin_dists))]

    pred_index = gt_idx_to_pred_idx_non_conv[gt_idx]
    pred_text = new_pred_texts[pred_index]
    gt_text = new_gt_texts[gt_idx]
    
    delete_gt_char_idxes = []
    for gt_char_idx, gt_char in enumerate(gt_text):
        # The reading direction is assumed to be from left to right.

        if pred_text.count(gt_char) == 1:
            delete_pred_char_idx = pred_text.index(gt_char)
            # remove a specific character(gt text) of prediction text.
            pred_text = "".join([ x for pred_char_idx, x in enumerate(pred_text) if pred_char_idx != delete_pred_char_idx])
            new_pred_texts[pred_index] = pred_text
            delete_gt_char_idxes.append(gt_char_idx)

        elif pred_text.count(gt_char) >= 2 and len(gt_char) != 0:
            ########################################################################
            ### if we can find two or more characters to be removed, we will remove left-most one.
            ### because we assumed the reading direction to be from left to right.
            ########################################################################
            delete_pred_char_idx_candidates = [idx for idx, c in enumerate(pred_text) if c == gt_char]
            delete_pred_char_idx = min(delete_pred_char_idx_candidates)

            # remove a specific character(gt text) of prediction text.
            pred_text = "".join([ x for pred_char_idx, x in enumerate(pred_text) if pred_char_idx != delete_pred_char_idx  ])
            new_pred_texts[pred_index] = pred_text
            delete_gt_char_idxes.append(gt_char_idx)

        elif pred_text.count(gt_char) == 0:
            pass

    new_gt_text = "".join([x for new_gt_char_idx, x in enumerate(gt_text) if new_gt_char_idx not in delete_gt_char_idxes])
    new_gt_texts[gt_idx] = new_gt_text
    if len(new_gt_text) == 0:
        new_gt_texts = [ c for idx, c in enumerate(new_gt_texts) if idx != gt_idx ]
        new_gt_boxes = [ c for idx, c in enumerate(new_gt_boxes) if idx != gt_idx ]


    return removeNoncontroversialBox(new_gt_boxes, new_gt_texts, pred_boxes, new_pred_texts)


def removeControversialBox(gt_boxes, gt_texts, pred_boxes, pred_texts, gt_box_del_candidates):
    gt_boxes, gt_texts, pred_boxes, pred_texts, gt_box_del_candidates = removeNoncontroversialBox(gt_boxes, gt_texts, pred_boxes, pred_texts)

    # if no gtbox remains, return the results.
    value_uniq = np.unique(list(gt_box_del_candidates.values()))
    if len(value_uniq) == 1:
        if len(value_uniq[0]) == 0:
            return gt_boxes, gt_texts, pred_boxes, pred_texts
    elif len(value_uniq) == 0:
        return gt_boxes, gt_texts, pred_boxes, pred_texts

    gt_idx_to_pred_idx_conv = {}
    for k in list(gt_box_del_candidates.keys()):
        if len(gt_box_del_candidates[k]) >= 2:
            gt_idx_to_pred_idx_conv[k] = gt_box_del_candidates[k]

    keys = list(gt_idx_to_pred_idx_conv.keys())
    gt_box_origin_dists = []
    for gt_idx in keys:
        gt_box = gt_boxes[gt_idx]
        gt_box_origin_dist = Point(0,0).distance(gt_box.representative_point())
        gt_box_origin_dists.append(gt_box_origin_dist)
    gt_idx = keys[gt_box_origin_dists.index(min(gt_box_origin_dists))]

    ################################################
    ## pred_candidates picking priority : 
    ## intersection area / gt_box area (bigger is earlier)
    ################################################
    pred_candidates = gt_idx_to_pred_idx_conv[gt_idx]
    ir_gt_box = gt_boxes[gt_idx]
    ir_pred_boxes = [pred_boxes[pred_idx] for pred_idx in pred_candidates]

    ir_list = []
    for ir_pred_box in ir_pred_boxes:
        ir = float(ir_pred_box.intersection(ir_gt_box).area) / float(ir_gt_box.area)
        ir_list.append(ir)
    ir_max = max(ir_list)
    ir_max_pred_box_idx = [i for i,j in enumerate(ir_list) if j == ir_max]

    #copy list
    new_pred_texts = list(pred_texts)
    new_gt_texts = list(gt_texts)
    new_gt_boxes = list(gt_boxes)

    delete_gt_idxes = []
    for ir_idx in ir_max_pred_box_idx:
        pred_index = pred_candidates[ir_idx]
        pred_text = new_pred_texts[pred_index]
        gt_text = new_gt_texts[gt_idx]

        delete_gt_char_idxes = []
        for gt_char_idx, gt_char in enumerate(gt_text):
            # left to right reading direction
            if pred_text.count(gt_char) == 1:
                delete_pred_char_idx = pred_text.index(gt_char)
                # remove a specific character(gt text) of prediction text.
                pred_text = "".join([ x for pred_char_idx, x in enumerate(pred_text) if pred_char_idx != delete_pred_char_idx  ])
                new_pred_texts[pred_index] = pred_text
                delete_gt_char_idxes.append(gt_char_idx)


            elif pred_text.count(gt_char) >= 2 and len(gt_char) != 0:
                # which one will be removed first? lefter is earlier.
                delete_pred_char_idx_candidates = [idx for idx, c in enumerate(pred_text) if c == gt_char]
                delete_pred_char_idx = min(delete_pred_char_idx_candidates)

                # remove a specific character(gt text) of prediction text.
                pred_text = "".join([ x for pred_char_idx, x in enumerate(pred_text) if pred_char_idx != delete_pred_char_idx  ])
                new_pred_texts[pred_index] = pred_text
                delete_gt_char_idxes.append(gt_char_idx)

            elif pred_text.count(gt_char) == 0:
                # no more same characters
                pass


        new_gt_text = "".join([x for new_gt_char_idx, x in enumerate(gt_text) if new_gt_char_idx not in delete_gt_char_idxes])
        new_gt_texts[gt_idx] = new_gt_text
        if len(new_gt_text) == 0:
            delete_gt_idxes.append(gt_idx)

    # removal
    new_gt_texts = [ c for idx, c in enumerate(new_gt_texts) if idx not in delete_gt_idxes ]
    new_gt_boxes = [ c for idx, c in enumerate(new_gt_boxes) if idx not in delete_gt_idxes ]

    gt_boxes, gt_texts, pred_boxes, pred_texts, gt_box_del_candidates = removeNoncontroversialBox(new_gt_boxes, new_gt_texts, pred_boxes, new_pred_texts)
    # tail recursion 
    return removeControversialBox(gt_boxes, gt_texts, pred_boxes, pred_texts, gt_box_del_candidates)

@profile
def papagoEval(gt_files, pred_files, dontcare_text=None):
    removed_gt_char_count = 0
    filename_to_f_score = {}

    precision_list = []
    recall_list = []

    total_removed_gt_char_count= 0
    total_pred_char_count = 0
    total_gt_chars_count = 0
    for i in range(len(pred_files)):
        if i % 100 == 0:
            print("%d / %d" % (i, len(pred_files)))
        filename = pred_files[i]
        pred_file = pred_files[i]
        gt_file = gt_files[i]
        try:
            with io.open(gt_file, 'r', encoding='utf-8') as f:
                gt_raw = f.read()
            with io.open(pred_file, 'r', encoding='utf-8') as f:
                pred_raw = f.read()

            #parsing GT from raw text
            ground_truths = [x for x in gt_raw.split("\n") if len(x) != 0]
            gt_boxes = []
            gt_texts = []
            for idx, anno in enumerate(ground_truths):
                # NOTICE: modify below line to make this work on multivertex polygons
                tokens = anno.split(SPLIT_DELIMITER)
                
                # temp
                gt_text = tokens[0].split(' ')[-8]
                # temp start

                # should be recovery
                # gt_text = tokens[1]
                # should be recovery

                # NOTICE: modify below line to make this work on multivertex polygons
                gt_box = getPolygon(chunker(tokens[0].split(" ")[:8], 2))
                if len(gt_text) != 0:
                    gt_boxes.append(gt_box)
                    gt_texts.append(gt_text)

            #parsing prediction from raw text
            predictions = [x for x in pred_raw.split("\n") if len(x) != 0]
            pred_boxes = []
            pred_texts = []
            for idx, pred in enumerate(predictions):
                # NOTICE: modify below line to make this work on multivertex polygons
                tokens = pred.split(SPLIT_DELIMITER)

                # temp
                pred_text = tokens[0].split(' ')[-8]
                # temp start

                # should be recovery
                # pred_text = tokens[1]
                # shoule be recovery

                # NOTICE: modify below line to make this work on multivertex polygons
                pred_box = getPolygon(chunker(tokens[0].split(" ")[:8], 2))
                pred_boxes.append(pred_box)
                pred_texts.append(pred_text)
            gt_boxes, gt_texts, pred_boxes, pred_texts = removeDoncareBox(gt_boxes, gt_texts, pred_boxes, pred_texts, dontcare_text) 
            step1 = removeNoncontroversialBox(gt_boxes, gt_texts, pred_boxes, pred_texts)
            step2 = removeControversialBox(*step1)
            
            unremoved_gt_boxes = step2[0]
            unremoved_gt_chars = step2[1]
            remain_pred_texts = step2[3]
            gt_char_count = len("".join(gt_texts)) 
            unremoved_gt_char_count = len("".join(unremoved_gt_chars))
            removed_gt_char_count = gt_char_count - unremoved_gt_char_count
            pred_char_count = len("".join(pred_texts))

            total_removed_gt_char_count += removed_gt_char_count
            total_pred_char_count += pred_char_count
            total_gt_chars_count += gt_char_count
            # Precision : removed character(among prediction) number / predicted character number
            #             removed_gt_char_count  / pred_char_count
            # Recall    : removed character(among prediction) number / ground truth character number
            #             removed_gt_char_count  / len(gt_boxes)
            if gt_char_count == 0:
                recall = float(1)
                precision = float(0) if pred_char_count > 0 else float(1)
            else:
                precision = _divide(float(removed_gt_char_count), float(pred_char_count))
                recall = _divide((removed_gt_char_count), float(gt_char_count))

            precision_list.append(precision)
            recall_list.append(recall)
        except Exception as e:
            print(filename, e)
            print(traceback.format_exc())


    precision_for_char = _divide(float(total_removed_gt_char_count), float(total_pred_char_count))
    recall_for_char = _divide(float(total_removed_gt_char_count), float(total_gt_chars_count))
    precision_avr = _divide(reduce(lambda x, y: x + y, precision_list, 0), len(precision_list))
    recall_avr = _divide(reduce(lambda x, y: x + y, recall_list, 0), len(recall_list))
    perf = _divide(2*(precision_for_char*recall_for_char), (precision_for_char + recall_for_char) )
    #print('| precision for entire char | average precision | recall for entire char | averate recall |')
    #print('| {} | {} | {} | {} |'.format(precision_for_char, precision_avr, recall_for_char, recall_avr ))
    #print("======================")
    return precision_for_char, recall_for_char, perf

def make_pair(GTs, Ds):
    if len(GTs)==0 or len(Ds)==0:
        return [], []
    
    def fname(x):
        _, fn = os.path.split(x)
        return fn

    newgts = []
    newds = []

    gi = di = 0
    while True:
        gtn = fname(GTs[gi])
        dn = fname(Ds[di])
        if gtn == dn:
            newgts += [GTs[gi]]
            newds += [Ds[di]]
            gi += 1
            di += 1
        elif gtn > dn:
            di += 1
        else:
            gi += 1

        if gi >= len(GTs) or di >= len(Ds):
            break

    assert len(newgts) == len(newds)
    return newgts, newds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtpath', required=True, help='directory of ground truth files')
    parser.add_argument('--dtpath', required=True, help='directory of prediction files')
    args = parser.parse_args()

    assert os.path.isdir(args.gtpath)
    assert os.path.isdir(args.dtpath)

    GT_files = sorted(glob.glob('%s/*.txt'%args.gtpath))
    D_files = sorted(glob.glob('%s/*.txt'%args.dtpath))

    if len(GT_files) != len(D_files):
        print("Caution: GT_files' len(%d) and D_files' len(%d) are different."%(len(GT_files), len(D_files)))
        GT_files, D_files = make_pair(GT_files, D_files)
        print("We will evaluate on %d files"%(len(GT_files)))
    
    pr, re, pref = papagoEval(GT_files, D_files, dontcare_text="###")
    print("precision, recall, H:")
    print("%0.1f, %0.1f, %0.1f"%(100.*pr, 100.*re, 100.*pref))

