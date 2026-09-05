import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./data/youzi_5/youzi_5/images"
output_dir = './data/youzi_5/youzi_5/object_mask'
output_filtered_dir = './data/youzi_5/youzi_5/filtered'
os.makedirs(output_filtered_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
# Initialize variables
points = []
labels = []

# Function to handle mouse clicks
def onclick(event):
    if event.inaxes:  # Check if the click is inside the axes
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # Left mouse button
            points.append([x, y])
            labels.append(1)
            plt.plot(x, y, 'ro')  # Red dot for positive click
        elif event.button == 3:  # Right mouse button
            points.append([x, y])
            labels.append(0)
            plt.plot(x, y, 'bo')  # Blue dot for negative click
        plt.draw()

# Load and display the image
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
plt.imshow(img)

# Connect the click event
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Convert lists to numpy arrays
points = np.array(points, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Debugging and further processing
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# ann_frame_idx and ann_obj_id can be used for further annotation logic
ann_frame_idx = 0
ann_obj_id = 1

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
# save

for out_obj_id, out_mask in video_segments.items():

    # Convert boolean mask to 0 and 255
    bw_image = (out_mask[1].astype(np.uint8)) * 255

    # Create an image from the array
    img = Image.fromarray(bw_image.squeeze())
    out_obj_id = int(out_obj_id)+1
    # Construct the filename
    filename = f"{str(out_obj_id).zfill(4)}.png"
    
    # Save the image
    img.save(os.path.join(output_dir, filename))

for frame_idx, frame_data in video_segments.items():
    # 加载原始图像
    original_image = np.array(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    for out_obj_id, mask in frame_data.items():
        # 将 mask 去掉第一个维度，变为 (100, 100)
        mask = mask.squeeze(0)  # 从 (1, 100, 100) 转为 (100, 100)

        # 将 mask 扩展为 (100, 100, 3)，以便与原图对齐
        mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

        # 创建一个与原图相同的空白图片
        filtered_image = np.zeros_like(original_image, dtype=np.uint8)

        # 将原图中 mask 为 True 的像素保留下来
        filtered_image[mask > 0] = original_image[mask > 0]

        # 保存结果图像
        filtered_image = Image.fromarray(filtered_image)
        filtered_filename = f"{str(frame_idx).zfill(4)}.jpg"
        filtered_image.save(os.path.join(output_filtered_dir, filtered_filename))

print(f"Filtered images saved to {output_filtered_dir}")
