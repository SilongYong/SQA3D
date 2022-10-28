import bpy
from mathutils import Euler, Vector, Matrix
import numpy as np
import json
import os

modes = ["BEV"]

data_version = "balanced"
splits = ["train", 'val', 'test']
scene_dir = "../assets/data/scannet/scans"              # TODO: change this
anno_dir = "../assets/data/sqa_task"                    # TODO: change this
anno_dir = os.path.join(anno_dir, data_version)
output_dir = "../assets/data/"                          # TODO: change this
s2qid = {}
scene_processed = []
for split in splits:
    questions = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_questions_{split}_scannetv2.json'), 'r'))['questions']
    annotations = json.load(open(os.path.join(anno_dir, f'v1_{data_version}_sqa_annotations_{split}_scannetv2.json'), 'r'))['annotations']
    qid2annoid = {}
    for i in range(len(annotations)):
        qid2annoid[annotations[i]["question_id"]] = i    
    scene_name_to_q_id = {}
    q_id_to_idx = {}
    for i in range(len(questions)):
        scene_name = questions[i]["scene_id"]
        q_id = questions[i]["question_id"]
        if scene_name not in scene_name_to_q_id.keys():
            scene_name_to_q_id[scene_name] = []
        scene_name_to_q_id[scene_name].append(q_id)
        q_id_to_idx[q_id] = i
    print(len(list(scene_name_to_q_id.keys())))
    for scene_name in scene_name_to_q_id.keys():
        ## LOAD SCENE
        
        bpy.ops.import_mesh.ply(filepath=os.path.join(scene_dir, scene_name, scene_name + '_vh_clean_2.ply'))
        obj = bpy.context.view_layer.objects.active
        material = bpy.data.materials['Material.001']
        obj.data.materials.append(material)

        points_co_global = []
        points_co_global.extend([obj.matrix_world @ vertex.co for vertex in obj.data.vertices])
        x, y, z = [[point_co[i] for point_co in points_co_global] for i in range(3)]
        def get_center(l):
            return (max(l) + min(l)) / 2 if l else 0.0
        b_sphere_center = Vector([get_center(axis) for axis in [x, y, z]]) if (x and y and z) else None
        obj.location.x -= b_sphere_center[0]
        obj.location.y -= b_sphere_center[1]
        obj.location.z -= b_sphere_center[2]
        radius = ((max(x) + max(y) + max(z)) - (min(x) + min(y) + min(z))) / 6
        
        q_id_list = scene_name_to_q_id[scene_name]
        for q_id in q_id_list:
            idx = q_id_to_idx[q_id]
            s = questions[idx]["situation"]
            if s in s2qid.keys():
                s2qid[s].append(q_id)
                continue
            s2qid[s] = [q_id]
            pos = [v for (k, v) in annotations[qid2annoid[q_id]]["position"].items()]
            rot = [v for [k, v] in annotations[qid2annoid[q_id]]["rotation"].items()]
            rot.insert(0, rot.pop())

             ## LOAD CAMERA
            MAT = Euler((0, np.pi / 2, 0)).to_matrix().to_4x4()
            MAT_Z = Euler((0, 0, np.pi / 2)).to_matrix().to_4x4()
            MAT_MOVE_Z = Matrix().to_4x4()
            MAT_MOVE_Z[2][3] = 1
            LOCATION = (pos[0], pos[1], pos[2])
            ROTATION = (rot[0], rot[1], rot[2], rot[3])

            cam = bpy.data.objects['Camera']
            cam.rotation_mode = 'QUATERNION'
            for mode in modes: 
                if mode == "BEV":
                    if scene_name in scene_processed:
                        continue
                    if os.path.exists(os.path.join(output_dir, mode, str(scene_name) + f"_{mode}.png")):
                        continue
                    bpy.context.scene.render.resolution_x = 1920
                    bpy.context.scene.render.resolution_y = 1080
                    camera_radius = 7 * radius
                    cam.location = (0, 0, camera_radius)
                    ## Setting
                    cam.data.lens = 45
                    cam.data.clip_start = 0.1
                    cam.data.clip_end = 1000

                    cam.rotation_quaternion = (1, 0, 0, 0)
                    bpy.context.view_layer.update()
                    ## RENDER
                    if not os.path.exists(os.path.join(output_dir, mode)):
                        os.mkdir(os.path.join(output_dir, mode))
                    bpy.context.scene.render.filepath = os.path.join(output_dir, mode, str(scene_name) + f"_{mode}.png")
                    bpy.ops.render.render(write_still = True)
            
            if scene_name not in scene_processed:
                scene_processed.append(scene_name)        
        bpy.data.objects.remove(obj)
        




