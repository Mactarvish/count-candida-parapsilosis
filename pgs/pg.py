import os
import glob
import cv2
import numpy as np


def filter_medium_region_by_hough_circle(src_image_np):
    h, w = src_image_np.shape[:2]
    dst_h, dst_w = int(384 * h / w), 384
    scale = w / dst_w
    # 宽度调整到384，保持横纵比不变
    src_image_np = cv2.resize(src_image_np, (dst_w, dst_h))
    # src_image_np = cv2.resize(src_image_np, (w//8, h//8))
    gray = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, dst_h / 2,
    param1=100, param2=30,
    minRadius=int(dst_w // 4), maxRadius=int(dst_w // 1.8))

    if circles is None:
        return None
    
    # 只考虑找到的第一个圆
    cx, cy, r = circles[0, 0]
    # 放缩回原图尺寸
    cx, cy, r = int(cx * scale), int(cy * scale), int(r * scale)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    
    return mask

    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(src_image_np, center, 1, (0, 100, 100), 1)
        # circle outline
        radius = i[2]
        cv2.circle(src_image_np, center, radius, (255, 0, 255), 1)
        


src_image_paths = glob.glob(os.path.join("pgs/src/*.jpg"))
for src_image_path in src_image_paths:
    src_image_np = cv2.imread(src_image_path)

    mask = filter_medium_region_by_hough_circle(src_image_np)
    cv2.imshow("mask", mask)
    cv2.imshow("src_image_np", src_image_np)
    cv2.waitKey(0)
    continue


    h, w = src_image_np.shape[:2]
    dst_h, dst_w = int(384 * h / w), 384
    # 宽度调整到384，保持横纵比不变
    src_image_np = cv2.resize(src_image_np, (dst_w, dst_h))
    # src_image_np = cv2.resize(src_image_np, (w//8, h//8))
    gray = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2GRAY)
    
    
    gray = cv2.medianBlur(gray, 3)
    
    
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 2,
    param1=100, param2=30,
    minRadius=int(dst_w // 4), maxRadius=int(dst_w // 1.8))
    
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
    else:
        print("啥也没有")
        continue
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(src_image_np, center, 1, (0, 100, 100), 1)
        # circle outline
        radius = i[2]
        cv2.circle(src_image_np, center, radius, (255, 0, 255), 1)
        
    cv2.imshow("detected circles", src_image_np)
    cv2.imshow("gray", gray)
    cv2.imshow('Canny Edges', edges)

    cv2.waitKey(0)
    