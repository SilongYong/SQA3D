import os
import cv2
from tqdm import tqdm
import glob
import argparse
from PIL import Image

def main(filepath, size, output_path):
    output_fn = os.path.split(filepath)[-1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(os.path.join(output_path, output_fn + ".mp4"), fourcc, 20, size)
    fl = sorted(glob.glob(os.path.join(filepath, "*.jpg")))
    for filename in tqdm([os.path.join(filepath, f"frame-{i}.color.jpg")for i in range(len(fl))]):
        img = cv2.resize(cv2.imread(filename), size)
        videowrite.write(img)
    videowrite.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default="../assets/data/Video")
    parser.add_argument('--file', default="../assets/data/Video_img")
    args = parser.parse_args()
    size = (224, 224)
    for folder in os.listdir(args.file):
        filepath = os.path.join(args.file, folder)
        main(filepath, size, args.output)
        
