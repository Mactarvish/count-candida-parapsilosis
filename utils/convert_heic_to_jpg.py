import numpy as np
import cv2
from tqdm import tqdm
import pillow_heif
import os
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    args = parser.parse_args()
    
    src_heic_paths = glob.glob(os.path.join(args.src_dir, "**", "*.heic"), recursive=True)
    for src_heic_path in tqdm(src_heic_paths):
        src_heic_np = np.array(pillow_heif.open_heif(src_heic_path, convert_hdr_to_8bit=False, bgr_mode=True))
        dst_jpg_path = src_heic_path.replace(".heic", ".jpg")
        cv2.imwrite(dst_jpg_path, src_heic_np)