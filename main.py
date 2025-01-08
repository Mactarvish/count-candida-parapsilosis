import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import sys

is_debug = sys.gettrace()

# 返回值类型检查，确保返回值是HW的uint8图像或None，避免蛋疼的类型错误
def assert_HW_0_255_or_None(f):
    def w(*args, **kwargs):
        r = f(*args, **kwargs)
        assert r is None or (len(r.shape) == 2 and r.dtype == np.uint8 and r.min() in [0, 255] and r.max() in [0, 255])
        return r
    return w


@assert_HW_0_255_or_None
def filter_medium_region_by_hough_circle(src_image_np, circle_info=None, color_mask=None):
    h, w = src_image_np.shape[:2]
    dst_h, dst_w = int(384 * h / w), 384
    scale = w / dst_w
    # 宽度调整到384，保持横纵比不变
    src_image_np = cv2.resize(src_image_np, (dst_w, dst_h))
    gray = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, dst_h / 6,
    param1=100, param2=30,
    minRadius=int(dst_w // 4), maxRadius=int(dst_w // 1.8))

    if circles is None:
        return None
    
    if color_mask is not None:
        color_mask = cv2.resize(color_mask, (dst_w, dst_h))
        color_mask[color_mask > 0] = 255

    # 只考虑找到的第一个圆
    best_cxcyr = None
    max_iou = 0
    oor_thr = 20
    for xyr in circles[0]:
        cx, cy, r = xyr
        cx, cy, r = int(cx), int(cy), int(r)
        # 过滤越界太多的圆
        if cx - r + oor_thr <= 0 or cy - r + oor_thr <= 0 or \
           cx + r - oor_thr >= dst_w or cy + r - oor_thr >= dst_h:
            continue
        mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        iou = np.sum(mask & color_mask) / np.sum(mask | color_mask)
        if iou > max_iou:
            max_iou = iou
            best_cxcyr = (cx, cy, r)

    # 放缩回原图尺寸
    if best_cxcyr is None:
        return None

    cx, cy, r = best_cxcyr
    cx, cy, r = int(cx * scale), int(cy * scale), int(r * scale)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    if circle_info is not None:
        assert isinstance(circle_info, list) and len(circle_info) == 0
        circle_info.extend([cx, cy, r])
    return mask


# 通过hsv阈值过滤出培养皿区域
@assert_HW_0_255_or_None
def filter_medium_region_by_hsv(src_image_np):
    src_image_hsv = cv2.cvtColor(src_image_np,cv2.COLOR_BGR2HSV)
    h_min = 0
    h_max = 40
    s_min = 50
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

    return mask

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
    # 只有背景，无有效前景
    if n == 1:
        return mask_np
    mask_np[labels == i_area_pair[0][0]] = True

    return mask_np



CALIBRATION_AREA = 600 * 800

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    if is_debug:
        args = parser.parse_args(["pgs/20240415coculture/0.2-B10.jpg"])
    else:
        args = parser.parse_args()

    if os.path.isfile(args.src_dir):
        src_image_paths = [args.src_dir]
        args.src_dir = os.path.dirname(args.src_dir)
    else:
        src_image_paths = glob.glob(os.path.join(args.src_dir, "**", "*.jpg"), recursive=True)
    f = open("count.txt", 'w')
    # 获取当前操作系统的路径分隔符
    for src_image_path in tqdm(src_image_paths):
        src_image_name = os.path.basename(src_image_path)
        print(src_image_path)
        dst_image_path = src_image_path.replace(os.path.basename(args.src_dir.rstrip(os.path.sep)), "dst")
        print(dst_image_path)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        src_image_np = cv2.imread(src_image_path).copy()
        print(src_image_np.shape)
        src_image_area = src_image_np.shape[0] * src_image_np.shape[1]
        h, w = src_image_np.shape[:2]

        # 获得培养皿区域mask
        # 先用颜色阈值给出一个培养皿区域proposal
        color_mask = filter_medium_region_by_hsv(src_image_np)
        circle_info = []
        # 用霍夫圆变换找出圆形培养皿，由于可能有多个结果，取和颜色proposal具有最大交集的那个，从而过滤掉误识
        mask = filter_medium_region_by_hough_circle(src_image_np, circle_info=circle_info, color_mask=color_mask)
        print(circle_info)

        infer_image_np = cv2.bitwise_and(src_image_np, src_image_np, mask=mask)
        medium_np = infer_image_np.copy()
        infer_image_hsv = cv2.cvtColor(infer_image_np,cv2.COLOR_BGR2HSV)

        # 通过饱和度阈值和value阈值过滤出培养皿中的白点
        h_min = 0
        h_max = 179
        s_min = 0
        s_max = 100
        v_min = 200
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
        tiny_count = 0
        small_count = 0
        small_areas = []
        big_count = 0
        for s in stats:
            x, y, w, h, area = s
            # 选出符合面积条件的白点
            # 滤除长条状的目标
            if (w / h > 3 or h / w > 3):
                continue
            # 滤除太大的目标
            if w * h / src_image_area > 20000 / (4032 * 3024):
                continue
            # 滤除mask面积相对外接矩形占比过小的目标
            # if area / (w * h) < 0.5:
                # continue
            if area / src_image_area < 5 / CALIBRATION_AREA:# and w * h / src_image_area < 120 / CALIBRATION_AREA:
                # 小号菌落
                if area > 1 * src_image_area / CALIBRATION_AREA:
                    cv2.rectangle(src_image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    small_count += 1
                    small_areas.append(area)
                else:
                    cv2.rectangle(src_image_np, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    tiny_count += 1
        
        # 统计小菌落平均面积
        mean_small_area = 0
        if len(small_areas) > 0:
            mean_small_area = float(np.mean(small_areas))

        for s in stats:
            x, y, w, h, area = s
            # 选出符合面积条件的白点
            # 滤除长条状的目标
            if (w / h > 3 or h / w > 3):
                continue
            # 滤除太大的目标
            if w * h / src_image_area > 20000 / (4032 * 3024):
                continue
            # 滤除mask面积相对外接矩形占比过小的目标
            # if area / (w * h) < 0.5:
                # continue
            # 大号菌落
            if not (area / src_image_area < 5 / CALIBRATION_AREA):# and w * h / src_image_area < 120 / CALIBRATION_AREA:
                count = int(np.round(area / mean_small_area))
                cv2.rectangle(src_image_np, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(src_image_np, str(count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                big_count += count
                

        print(small_count)
        cv2.putText(src_image_np, "tiny: " + str(tiny_count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        cv2.putText(src_image_np, "small: " + str(small_count), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv2.putText(src_image_np, "big (multipled by valid area times): " + str(big_count), (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(src_image_np, "summary: " + str(tiny_count + small_count + big_count), (100, 700), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 8)
        f.write(src_image_path + " " + str(small_count) + " " + str(big_count) + "\n")
        cv2.circle(src_image_np, (circle_info[0], circle_info[1]), circle_info[2], (255, 0, 255), 6)
        cv2.imwrite(dst_image_path, src_image_np)
        
        if is_debug:
            show_size = (600, 800)
            cv2.imshow("medium", cv2.resize(medium_np, show_size))
            cv2.imshow("white dot", cv2.resize(white_dot_np, show_size))
            cv2.imshow("count", cv2.resize(src_image_np, show_size))
            # 三个图拼成一张图
            cv2.imwrite("pgs/vis_mid.jpg", np.concatenate([medium_np, white_dot_np, src_image_np], axis=1))
            cv2.waitKey(0)
            exit(0)
    f.close()
