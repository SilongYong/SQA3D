import json
import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def metric_localization(
    gt_pos,
    gt_rot,
    pred_pos,
    pred_rot,
):
    """
    gt_pos: [N, 3]; xyz
    gt_rot: [N, 4]; xyzw
    pred_pos: a list with N elements, each element is a list of pos predictions
    pred_rot: a list with N elements, each element is a list of rot predictions
    """
    def pos_distance(pos1, pos2):
        # ignore z
        return math.sqrt(sum((pos1[:2] - pos2[:2])**2))

    def rot_distance(rot1, rot2):
        # only consider rotation along z-axis, range is -pi~pi
        r1 = R.from_quat(rot1).as_rotvec()[-1]
        r2 = R.from_quat(rot2).as_rotvec()[-1]

        return min(abs(r1 - r2), 2 * math.pi - abs(r1 - r2)) / math.pi * 180

    cnt_pos_0_5, cnt_pos_1 = 0, 0
    cnt_rot_15, cnt_rot_30 = 0, 0
    for gt_p, gt_r, pred_p, pred_r in zip(gt_pos, gt_rot, pred_pos, pred_rot):
        posdiff = min([pos_distance(gt_p, p) for p in pred_p])
        rotdiff = min([rot_distance(gt_r, r) for r in pred_r])
        if posdiff < 0.5:
            cnt_pos_0_5 += 1
        if posdiff < 1:
            cnt_pos_1 += 1
        if rotdiff < 15:
            cnt_rot_15 += 1
        if rotdiff < 30:
            cnt_rot_30 += 1
    total = len(gt_pos)

    print(f"""
          Report:

          Position prediction:
          -Acc@0.5m: {cnt_pos_0_5/total}
          -Acc@1.0m: {cnt_pos_1/total}

          Rotation prediction:
          -Acc@15°: {cnt_rot_15/total}
          -Acc@30°: {cnt_rot_30/total}
          """)
    return {
        'acc@0.5m': {cnt_pos_0_5/total},
        'acc@1.0m': {cnt_pos_1/total},
        'acc@15°': {cnt_rot_15/total},
        'acc@30°': {cnt_rot_30/total},
    }


if __name__ == '__main__':
    gtlabel = json.load(open(
        'assets/data/sqa_task/balanced/v1_balanced_sqa_annotations_test_scannetv2.json', 'r'))['annotations']
    gt_pos, gt_rot, pred_pos, pred_rot = [], [], [], []
    for label in gtlabel:
        gt_pos.append([*label['position'].values()])
        gt_rot.append([*label['rotation'].values()])
        pred_pos.append([np.random.rand(3) for _ in range(3)])
        pred_rot.append([R.from_rotvec(
            [0, 0, np.random.rand()*math.pi*2-math.pi]).as_quat() for _ in range(3)])

    metric_localization(gt_pos, gt_rot, pred_pos, pred_rot)
