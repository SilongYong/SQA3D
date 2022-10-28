import time
import os
import subprocess
import argparse

def get_eta(start, end, extra, num_left):
    exe_s = end - start
    eta_s = (exe_s + extra) * num_left
    eta = {'h': 0, 'm': 0, 's': 0}
    if eta_s < 60:
        eta['s'] = int(eta_s)
    elif eta_s >= 60 and eta_s < 3600:
        eta['m'] = int(eta_s / 60)
        eta['s'] = int(eta_s % 60)
    else:
        eta['h'] = int(eta_s / (60 * 60))
        eta['m'] = int(eta_s % (60 * 60) / 60)
        eta['s'] = int(eta_s % (60 * 60) % 60)

    return eta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_all', type=str, help="temp_scene_folder", default="../assets/data/scannet/meta_data/scannetv2.txt")
    parser.add_argument('--meta_test', type=str, help="save_scene_folder", default="../assets/data/scannet/meta_data/scannetv2_test.txt")
    parser.add_argument('--output_folder', type=str, help="where to output the files", default="../assets/data/Video_img")
    parser.add_argument('--scan_file_path', type=str, help="where scan file is", default="../assets/data/scannet/scans")
    parser.add_argument('--sens_path', type=str, help="where sens are stored", default="/ScanNet_repo/SensReader/c++/sens")
    args = parser.parse_args()
    SCANNET_ALL_NAMES = sorted([line.rstrip() for line in open(args.meta_all) if "_00" in line])
    SCANNET_TEST_NAMES = sorted([line.rstrip() for line in open(args.meta_test) if "_00" in line])
    SCAN_NAMES = [scan_names for scan_names in SCANNET_ALL_NAMES if scan_names not in SCANNET_TEST_NAMES]
    OUTPUT_FOLDER = args.output_folder
    SCANS_PATH = args.scan_file_path
    SENS_PATH = args.sens_path
    all_scene = SCAN_NAMES
    for i, scene in enumerate(all_scene):
        if not os.path.exists(os.path.join(OUTPUT_FOLDER, scene)):
            os.mkdir(os.path.join(OUTPUT_FOLDER, scene))
        start = time.time()
        scan_path = os.path.join(SCANS_PATH, scene)
        sens_file_name = scene + ".sens"
        sens_file_path = os.path.join(scan_path, sens_file_name)
        scene_output_dir = os.path.join(OUTPUT_FOLDER, scene + "_temp")
        if not os.path.exists(scene_output_dir):
            os.mkdir(scene_output_dir)
        cmd = [SENS_PATH, sens_file_path, scene_output_dir]
        _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        scene_output_final_dir = os.path.join(OUTPUT_FOLDER, scene)
        cmd = ["python", "downsample.py", "--temp_dir", scene_output_dir, "--save_dir", scene_output_final_dir]
        _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        cmd = ["rm", "-rf", scene_output_dir]
        _ = subprocess.call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        # verbose
        num_left = len(all_scene) - i - 1
        eta = get_eta(start, time.time(), 0, num_left)
        print("movie downsample on {}, {} scenes left, ETA: {}h {}m {}s".format(
            scene,
            num_left,
            eta["h"],
            eta["m"],
            eta["s"]
        ))
    print("done!")