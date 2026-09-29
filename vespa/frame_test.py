from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes', verbose=False)
scene = next(s for s in nusc.scene if s['name'] == 'scene-euro-citystrasbourg-scenariolatesession-00002_1')

token = nusc.get('sample', scene['first_sample_token'])['data']['LIDAR_TOP']
for _ in range(1330):
    token = nusc.get('sample_data', token)['next']

sd = nusc.get('sample_data', token)
while not sd['is_key_frame']:
    token = sd['next']
    sd = nusc.get('sample_data', token)

sample = nusc.get('sample', sd['sample_token'])
cam_sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
print(f"/media/lleba/ECP_Nuscenes_01/output2/ecp2nuscenes/{cam_sd['filename']}")