import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import sys

is_debug = sys.gettrace()

def keep_greater_than_n_size_mask(mask_np, min_area):
    '''
    计算mask中每个连通域的面积，只保留面积大于min_area的连通域
    :param mask_np:
    :param min_area:最小连通域面积阈值
    :return:
    '''
    assert mask_np.dtype == np.bool, mask_np.dtype
    mask_np = np.uint8(mask_np)
    # labels从0开始计
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
    i_area_pair = [(i, area) for i, area in zip(range(1, n), stats[1:, 4])]
    i_area_pair = sorted(i_area_pair, key=lambda p: p[1], reverse=True)

    mask_np = np.zeros(mask_np.shape, dtype=np.bool)
    for (i, area) in i_area_pair:
        if area < min_area:
            break
        mask_np[labels == i] = True

    return mask_np


def keep_max_area_mask(mask_np):
    '''
    返回最大面积的mask
    :param mask_np:
    :param min_area:最小连通域面积阈值
    :return:
    '''
    assert mask_np.dtype == bool, mask_np.dtype
    mask_np = np.uint8(mask_np)
    # labels从0开始计
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)
    i_area_pair = [(i, area) for i, area in zip(range(1, n), stats[1:, 4])]
    i_area_pair = sorted(i_area_pair, key=lambda p: p[1], reverse=True)

    mask_np = np.zeros(mask_np.shape, dtype=bool)
    mask_np[labels == i_area_pair[0][0]] = True

    return mask_np



CALIBRATION_AREA = 600 * 800

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    if is_debug:
        args = parser.parse_args([r"D:\fengyongronglu\Coculture\18A1.jpg"])
    else:
        args = parser.parse_args()

    if os.path.isfile(args.src_dir):
        src_image_paths = [args.src_dir]
    else:
        src_image_paths = glob.glob(os.path.join(args.src_dir, "**", "*.jpg"), recursive=True)

    for src_image_path in tqdm(src_image_paths):
        src_image_name = os.path.basename(src_image_path)
        print(src_image_path)
        dst_image_path = os.path.join(os.path.dirname(args.src_dir), "dst", src_image_name)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        src_image_np = cv2.imread(src_image_path).copy()
        print(src_image_np.shape)
        src_image_area = src_image_np.shape[0] * src_image_np.shape[1]
        h, w = src_image_np.shape[:2]
        # 通过hsv阈值过滤出培养皿区域
        src_image_hsv = cv2.cvtColor(src_image_np,cv2.COLOR_BGR2HSV)
        h_min = 0
        h_max = 21
        s_min = 100
        s_max = 255
        v_min = 0
        v_max = 255

        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(src_image_hsv,lower,upper).astype(bool)
        # 保留最大面积的连通域
        mask = keep_max_area_mask(mask).astype(np.uint8) * 255
        # 将连通域内的空洞填充，作为培养皿区域
        kernel = np.ones((51, 51), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        infer_image_np = cv2.bitwise_and(src_image_np, src_image_np, mask=mask)
        medium_np = infer_image_np.copy()
        infer_image_hsv = cv2.cvtColor(infer_image_np,cv2.COLOR_BGR2HSV)

        # 通过饱和度阈值过滤出培养皿中的白点
        h_min = 0
        h_max = 179
        s_min = 0
        s_max = 255
        s_max = 80
        v_min = 0
        v_max = 255
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(infer_image_hsv,lower,upper) & mask
        # 只保留白点区域
        infer_image_np = cv2.bitwise_and(infer_image_np, infer_image_np, mask=mask)
        white_dot_np = infer_image_np.copy()

        # 获取每个白点（连通域）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        # stats按照面积排序
        resort =stats[:, 4].argsort() 
        stats = stats[resort]
        # resort_map = dict()
        # for i, r in enumerate(resort):
        #     resort_map[i] = -r
        
        # labels = resort_map[labels]
        # valid_indexes_mask = (stats[:, -1] < 100) & (stats[:, -1] > 1)
        # num_labels = np.sum(valid_indexes_mask)
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         if valid_indexes_mask[labels[i, j]]:
        #             infer_image_np[i, j] = [0, 0, 255]
        
        # infer_image_np[labels != 0] = [0, 0, 255]
        small_count = 0
        big_count = 0
        for s in stats:
            x, y, w, h, area = s
            # 选出符合面积条件的白点
            if (w / h > 3 or h / w > 3):
                continue
            if w * h / src_image_area > 20000 / (4032 * 3024):
                continue
            if area / src_image_area < 300 / CALIBRATION_AREA and w * h / src_image_area < 120 / CALIBRATION_AREA:
                # 小号菌落
                if area > 1 * src_image_area / CALIBRATION_AREA:
                    cv2.rectangle(src_image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    small_count += 1
            else:
                # 大号菌落
                cv2.rectangle(src_image_np, (x, y), (x + w, y + h), (0, 0, 255), 3)
                big_count += 1
                

        print(small_count)
        cv2.putText(src_image_np, "small: " + str(small_count), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
        cv2.putText(src_image_np, "big: " + str(big_count), (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        cv2.imwrite(dst_image_path, src_image_np)
        
        if is_debug:
            show_size = (600, 800)
            cv2.imshow("medium", cv2.resize(medium_np, show_size))
            cv2.imshow("white dot", cv2.resize(white_dot_np, show_size))
            cv2.imshow("count", cv2.resize(src_image_np, show_size))
            cv2.waitKey(0)
            exit(0)