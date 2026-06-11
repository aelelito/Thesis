import numpy as np

file_path = "/media/lleba/ECP_Nuscenes_01/SAM3_Visualizations/Strassbourg/scene_10_cam_0_sam3_outputs__car.npy"

data = np.load(file_path, allow_pickle=True).item()
frames = sorted(data.keys())

print(f"Total frames: {len(frames)}")

# ===== STEP 1: Collect raw frame lists per SAM3 ID =====
id_presence = {}

for frame_idx in frames:
    ids = data[frame_idx]['out_obj_ids']

    for obj_id in ids:
        if obj_id not in id_presence:
            id_presence[obj_id] = []
        id_presence[obj_id].append(frame_idx)

# ===== STEP 2: Split into continuous segments =====
def split_into_segments(frame_list):
    segments = []
    current_segment = [frame_list[0]]

    for i in range(1, len(frame_list)):
        if frame_list[i] == frame_list[i - 1] + 1:
            current_segment.append(frame_list[i])
        else:
            segments.append(current_segment)
            current_segment = [frame_list[i]]

    segments.append(current_segment)
    return segments

# ===== STEP 3: Assign NEW consistent IDs =====
new_id_counter = 0
consistent_tracks = {}   # new_id → frames

print("\n=== SPLIT TRACKS ===")

for obj_id in sorted(id_presence.keys()):
    frames_seen = sorted(id_presence[obj_id])
    segments = split_into_segments(frames_seen)

    print(f"\nOriginal ID {obj_id}: {len(segments)} segment(s)")

    for seg in segments:
        new_id = new_id_counter
        new_id_counter += 1

        consistent_tracks[new_id] = seg

        print(f"  → New ID {new_id}: frames {seg[0]} → {seg[-1]} (len={len(seg)})")

# ===== FINAL SUMMARY =====
print("\n=== FINAL CONSISTENT IDS ===")
for new_id, seg in consistent_tracks.items():
    print(f"ID {new_id}: frames {seg[0]} → {seg[-1]} (len={len(seg)})")

print(f"\nTotal unique objects (after split): {len(consistent_tracks)}")