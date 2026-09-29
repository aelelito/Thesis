"""Dry run of testing/sam3d_mask_freespace_analysis.ipynb on CPU: executes its code cells with a FAKE pipeline (GT-box meshes, nuScenes frames only).
    bash container/run_in_container.sh python tests/dry_run_freespace_notebook.py   (writes into results/sam3d_mask_freespace: delete those files afterwards)
"""
import sys, json, numpy as np, traceback
sys.path.insert(0, '/workspace/autolabeling/src')
import matplotlib; matplotlib.use('Agg')
import plotly.io as pio; pio.renderers.default = 'json'   # no browser in the dry run
from autolabeling import pilot
from autolabeling.utils import freespace_viz as fv, mesh_mask as mm

pilot.SELECTIONS = [s for s in pilot.SELECTIONS if s['dataset'] == 'nuscenes_mini']
F = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])

def fake_run(sel, nusc, cfg, ckpt_root, device=None):
    frame = pilot.get_frame(nusc, sel); img, _ = frame.load_images(); H, W = img.shape[:2]
    gts = pilot.gt_with_speed(nusc, frame); objs = []
    for g in gts:
        if not g['category'].startswith('vehicle.car'): continue
        pc = frame.R_c2e.T @ (g['center'] - frame.t_c2e)
        if pc[2] < 3: continue
        u = pc[0]/pc[2]*frame.K[0,0]+frame.K[0,2]; v = pc[1]/pc[2]*frame.K[1,1]+frame.K[1,2]
        if not (100 < u < W-100 and 100 < v < H-100): continue
        c, l, w, h, yaw = pilot.gt_box_of(g)
        L = l + 0.5                                   # fake overshoot: 0.5 m too long
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        cs, sn = np.cos(yaw), np.sin(yaw)
        Vw = np.stack([cs*V[:,0]*L/2 - sn*V[:,1]*w/2, sn*V[:,0]*L/2 + cs*V[:,1]*w/2, V[:,2]*h/2], 1) + c
        vc = ((Vw - frame.t_c2e) @ frame.R_c2e).astype(np.float32)
        sil, _ = mm.render_mesh(vc, F, frame.K, H, W)
        if sil.sum() < 500: continue
        m = sil.copy(); m[:, :m.shape[1]//2] &= True
        objs.append(dict(vertices=vc, faces=F.astype(np.int32), binary_mask=m, prompt='car', score=0.9, sam3_index=len(objs),
                         o3_mode='fake', obb_center=c, obb_dims=np.array([L, w, h]), obb_yaw=yaw,
                         obb_corners=fv.box_corners(c, L, w, h, yaw), obb_raw=dict(center=c, dims=np.array([L, w, h]), yaw=yaw)))
    return frame, objs
pilot.run_selection = fake_run

nb = json.load(open('/workspace/testing/sam3d_mask_freespace_analysis.ipynb'))
import IPython.display as ipd
g = {'__name__': '__main__'}
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source']).replace("RUN_TAGS = ['ecp10_f560']", "RUN_TAGS = ['nusc3_kf3']")
    print(f'--- cell {i}', src.strip().splitlines()[0][:70], flush=True)
    try:
        exec(compile(src, f'cell{i}', 'exec'), g)
    except Exception:
        traceback.print_exc(); break
import matplotlib.pyplot as plt; plt.close('all')
print('DONE')
