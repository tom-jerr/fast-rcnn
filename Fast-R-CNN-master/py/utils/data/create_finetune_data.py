# -*- coding: utf-8 -*-



import time
import shutil
import numpy as np
import cv2
import os
import selectivesearch
from util import check_dir
from util import parse_car_csv
from util import parse_xml
from util import compute_ious


# train
# positive num: 67139
# negative num: 427068
# val
# positive num: 65337
# negative num: 408609


def parse_annotation_jpeg(annotation_path, jpeg_path, gs):
    """
    获取正负样本（注：忽略属性difficult为True的标注边界框）
    正样本：候选建议与标注边界框IoU大于等于0.5 + 标注边界框
    负样本：IoU大于0.1,小于0.5
    """
    img = cv2.imread(jpeg_path)

    selectivesearch.config(gs, img, strategy='q')
    # 计算候选建议
    rects = selectivesearch.get_rects(gs)
    # 获取标注边界框
    bndboxs = parse_xml(annotation_path)

    # 获取候选建议和标注边界框的IoU
    iou_list = compute_ious(rects, bndboxs)

    positive_list = list()
    negative_list = list()
    for i in range(len(iou_list)):
        xmin, ymin, xmax, ymax = rects[i]
        rect_size = (ymax - ymin) * (xmax - xmin)

        iou_score = iou_list[i]
        if iou_score >= 0.5:
            # 正样本
            positive_list.append(rects[i])
        if 0.1 <= iou_score < 0.5:
            # 负样本
            negative_list.append(rects[i])
        else:
            pass

    # 添加标注边界框到正样本列表
    positive_list.extend(bndboxs)

    return positive_list, negative_list


if __name__ == '__main__':
    car_root_dir = '../../data/voc_car/'
    finetune_root_dir = '../../data/finetune_car/'
    check_dir(finetune_root_dir)

    gs = selectivesearch.get_selective_search()
    for name in ['train', 'val']:
        src_root_dir = os.path.join(car_root_dir, name)
        src_annotation_dir = os.path.join(src_root_dir, 'Annotations')
        src_jpeg_dir = os.path.join(src_root_dir, 'JPEGImages')

        dst_root_dir = os.path.join(finetune_root_dir, name)
        dst_annotation_dir = os.path.join(dst_root_dir, 'Annotations')
        dst_jpeg_dir = os.path.join(dst_root_dir, 'JPEGImages')
        check_dir(dst_root_dir)
        check_dir(dst_annotation_dir)
        check_dir(dst_jpeg_dir)

        total_num_positive = 0
        total_num_negative = 0

        samples = parse_car_csv(src_root_dir)
        # 复制csv文件
        src_csv_path = os.path.join(src_root_dir, 'car.csv')
        dst_csv_path = os.path.join(dst_root_dir, 'car.csv')
        shutil.copyfile(src_csv_path, dst_csv_path)
        for sample_name in samples:
            since = time.time()

            src_annotation_path = os.path.join(src_annotation_dir, sample_name + '.xml')
            src_jpeg_path = os.path.join(src_jpeg_dir, sample_name + '.jpg')
            # 获取正负样本
            positive_list, negative_list = parse_annotation_jpeg(src_annotation_path, src_jpeg_path, gs)
            total_num_positive += len(positive_list)
            total_num_negative += len(negative_list)
            dst_annotation_positive_path = os.path.join(dst_annotation_dir, sample_name + '_1' + '.csv')
            dst_annotation_negative_path = os.path.join(dst_annotation_dir, sample_name + '_0' + '.csv')
            dst_jpeg_path = os.path.join(dst_jpeg_dir, sample_name + '.jpg')
            # 保存图片
            shutil.copyfile(src_jpeg_path, dst_jpeg_path)
            # 保存正负样本标注
            np.savetxt(dst_annotation_positive_path, np.array(positive_list), fmt='%d', delimiter=' ')
            np.savetxt(dst_annotation_negative_path, np.array(negative_list), fmt='%d', delimiter=' ')

            time_elapsed = time.time() - since
            print('parse {}.png in {:.0f}m {:.0f}s'.format(sample_name, time_elapsed // 60, time_elapsed % 60))
        print('%s positive num: %d' % (name, total_num_positive))
        print('%s negative num: %d' % (name, total_num_negative))
    print('done')
