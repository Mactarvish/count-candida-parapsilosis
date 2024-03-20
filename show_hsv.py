import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 滑动条的回调函数，获取滑动条位置处的值
def visualize_hsv(a):
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Value Max", "TrackBars")

    # h_min,h_max,s_min,s_max,v_min,v_max = visualize_hsv(0)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    # 获得指定颜色范围内的掩码
    mask = cv2.inRange(imgHSV,lower,upper)
    # 对原图图像进行按位与的操作，掩码区域保留
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    return h_min, h_max, s_min, s_max, v_min, v_max


def gamma_rectify(img, gamma=1.2):
    # 将图像转换为浮点数格式
    img_float = img.astype(float)
    # Gamma校正
    img_gamma = img_float ** gamma
    # 调整数据范围
    img_gamma = (img_gamma.clip(0, 255)).astype(np.uint8)
    return img_gamma


def matplotlib2opencv(path):
    # 读取
    img = plt.imread(path)*255
    img = img.clip(0, 255)
    # 改变通道顺序
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    return img_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", nargs='+')
    args = parser.parse_args()
    print(args.src_dir)
    if len(args.src_dir) == 1:
        args.src_dir = args.src_dir[0]
        if os.path.isfile(args.src_dir):
            src_image_paths = [args.src_dir]
        else:
            src_image_paths = glob.glob(os.path.join(args.src_dir, "**", "*.jpg"), recursive=True)
    else:
        src_image_paths = args.src_dir
        
    src_image_nps = []
    for src_image_path in src_image_paths:
        print(src_image_path)
        src_image_np = cv2.imread(src_image_path)
        src_image_np = cv2.resize(src_image_np, (600, 800))
        h, w = src_image_np.shape[:2]
        # src_image_np = src_image_np[h//4:-h//4, w//4:-w//4, ...]
        src_image_nps.append(src_image_np)
    # h = h // 2
    # w = w // 2

    if len(src_image_nps) == 1:
        canvas_np = src_image_np
    else:
        num_images = len(src_image_nps)
        row = int(np.sqrt(num_images))
        for i in range(row, -1, -1):
            if num_images % i == 0:
                row = i
                col = num_images // i
                break
        canvas_np = np.zeros((h*row, w*col, 3), np.uint8)
        for i in range(row):
            for j in range(col):
                canvas_np[i*h:(i+1)*h, j*w:(j+1)*w, :] = src_image_nps[i*2 + j]

    img = canvas_np
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,240)
    cv2.createTrackbar("Hue Min","TrackBars",0,179,visualize_hsv)
    cv2.createTrackbar("Hue Max","TrackBars",21,179,visualize_hsv)
    cv2.createTrackbar("Sat Min","TrackBars",100,255,visualize_hsv)
    cv2.createTrackbar("Sat Max","TrackBars",255,255,visualize_hsv)
    cv2.createTrackbar("Value Min","TrackBars",0,255,visualize_hsv)
    cv2.createTrackbar("Value Max","TrackBars",255,255,visualize_hsv)

    while cv2.waitKey(0) != 27:
        ...