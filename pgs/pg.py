import os
import glob
import cv2

src_image_paths = glob.glob(os.path.join("pgs/src/*.jpg"))
for src_image_path in src_image_paths:
    print(src_image_path)
    src_image_np = cv2.imread(src_image_path).copy()
    src_image_np[:int(src_image_np.shape[0] * 4 / 10), ...] = 0

    cv2.imwrite(src_image_path, src_image_np)
