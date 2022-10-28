import os
import argparse
import glob
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_dir', type=str, help="temp_scene_folder")
    parser.add_argument('--save_dir', type=str, help="save_scene_folder")
    args = parser.parse_args()
    fl = sorted(glob.glob(os.path.join(args.temp_dir, "*.jpg")))
    new_fl = []
    for i in range(0, len(fl), 20):
        new_fl.append(fl[i])
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for i, filename in enumerate(new_fl):
        src = filename
        new_filename = f"frame-{i}.color.jpg"
        dest = os.path.join(args.save_dir, new_filename)
        shutil.move(src, dest)
    