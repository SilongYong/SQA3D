import trimesh as tm
import open3d as o3d
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', default="scene0000_00", help="visualize which scene, format: sceneXXXX_00", type=str)
    parser.add_argument('--anno_path', default="./assets/data/sqa_task/", type=str, help="annotation path, shoule be sqa_task directory")
    parser.add_argument('--ply_path', default='./ScanQA/data/scannet/scans', type=str, help="scene path, should be directory to original ScanNet scans")
    args = parser.parse_args()

    answer_dict = json.load(open(os.path.join(args.anno_path, "answer_dict.json"), 'r'))
    annotation_path = os.path.join(args.anno_path, "balanced")
    answer2class = answer_dict[0]
    class2answer = answer_dict[1]
    all_questions = \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_questions_train_scannetv2.json'), 'r'))['questions'] + \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_questions_val_scannetv2.json'), 'r'))['questions'] + \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_questions_test_scannetv2.json'), 'r'))['questions']
    all_annotations = \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_sqa_annotations_train_scannetv2.json'), 'r'))['annotations'] + \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_sqa_annotations_val_scannetv2.json'), 'r'))['annotations'] + \
        json.load(open(os.path.join(annotation_path, 'v1_balanced_sqa_annotations_test_scannetv2.json'), 'r'))['annotations']

    qid2annoid = {}
    for i in range(len(all_annotations)):
        qid2annoid[all_annotations[i]["question_id"]] = i

    for entry in all_questions:
        cone = tm.creation.cone(radius=0.1, height=0.20, sections=None, transform=None)
        rotate_around_y = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        cone = cone.apply_transform(rotate_around_y)
        cylinder = tm.creation.cylinder(radius=0.06, height=0.30, sections=None, segment=None, transform=None)
        cylinder = cylinder.apply_transform(rotate_around_y)
        mv_2_head = np.array([[1, 0, 0, -0.15], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        cone = cone.apply_transform(mv_2_head)
        arrow = tm.util.concatenate([cone, cylinder])

        scene_id = entry["scene_id"]
        situation = entry["situation"]
        if scene_id != args.scene_id:
            continue
        alter_situation = entry["alternative_situation"]
        question = entry["question"]
        question_id = entry["question_id"]
        ply_path = os.path.join(args.ply_path, scene_id, scene_id + "_vh_clean_2.ply")
        scene = tm.load(ply_path)
        v = np.array(scene.vertices)
        bs_center = (np.max(v[:, 0 : 3], axis=0) + np.min(v[:, 0 : 3], axis=0)) / 2
        scene_transformation = np.array([[1, 0, 0, -bs_center[0]],
                                        [0, 1, 0, -bs_center[1]],
                                        [0, 0, 1, -bs_center[2]],
                                        [0, 0, 0, 1]])
        scene = scene.apply_transform(scene_transformation)
        annotation = all_annotations[qid2annoid[question_id]]
        answer = annotation["answers"][0]["answer"]
        rotation = annotation["rotation"]  # {"_x": 0, "_y": 0, "_z": -0.9989515314916598, "_w": 0.04578032034039704}
        position = annotation["position"]  # {"x": -0.11717852338607089, "y": -0.21705917158968568, "z": 0}
        quaternion = [rotation["_x"], rotation["_y"], rotation["_z"], rotation["_w"]]
        temp = R.from_quat(quaternion)
        rot_mat_3x3 = temp.as_matrix()
        rotation_matrix = np.array([[rot_mat_3x3[0][0], rot_mat_3x3[0][1], rot_mat_3x3[0][2], 0],
                                    [rot_mat_3x3[1][0], rot_mat_3x3[1][1], rot_mat_3x3[1][2], 0],
                                    [rot_mat_3x3[2][0], rot_mat_3x3[2][1], rot_mat_3x3[2][2], 0],
                                    [0, 0, 0, 1]])
        transformation_matrix = np.array([[1, 0, 0, position["x"]],
                                        [0, 1, 0, position["y"]],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
        this_arrow = arrow.apply_transform(rotation_matrix)
        this_arrow = this_arrow.apply_transform(transformation_matrix)
        this_arrow.visual.vertex_colors = np.zeros((100, 4))
        this_arrow.visual.vertex_colors[:, 0] = 0
        this_arrow.visual.vertex_colors[:, 1] = 255
        this_arrow.visual.vertex_colors[:, 2] = 0
        this_arrow.visual.vertex_colors[:, 3] = 255

        whole_scene = tm.util.concatenate([scene, this_arrow])
        print("situation: ", situation)
        print("question: ", question)
        for i, sit in enumerate(alter_situation):
            print("alternative situation {i}: ", sit)
        whole_scene.show()
        # whole_scene.export(output_file)