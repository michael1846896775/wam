import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button

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
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

import cv2
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Process video directories and output paths")

# 创建解析器
parser = argparse.ArgumentParser(description="Process video directories and output paths")

# 添加命令行参数
parser.add_argument(
    '--video_dir', 
    type=str, 
    default="/extdatashare/dir_wangam00/rice1_pic",
    help="Root directory containing all video frame folders"
)

parser.add_argument(
    '--output_dir', 
    type=str, 
    default='/extdatashare/dir_wangam00/data',
    help="Root directory where the object masks will be saved"
)

# 解析命令行参数
args = parser.parse_args()

# 获取传入的参数
video_dir = args.video_dir
output_dir = args.output_dir

print(f"Video root directory: {video_dir}")
print(f"Output root directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# 获取所有视频帧文件夹
video_dirs = [os.path.join(video_dir, d) for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
video_dirs.sort()

# 当前视频索引
current_video_idx = 0
current_video_dir = os.path.join(video_dirs[current_video_idx], "images")  # 加载每个视频的 images 次级目录
frame_names = sorted(
    [p for p in os.listdir(current_video_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))],
    key=lambda p: int(os.path.splitext(p)[0]),
)

if not frame_names:
    print(f"No frames found in directory: {current_video_dir}")
    exit()

frame_idx = 0
current_image = Image.open(os.path.join(current_video_dir, frame_names[frame_idx]))



colors = [np.array([0, 128, 255], dtype=np.uint8),
         np.array([255, 128, 0], dtype=np.uint8),
         np.array([0, 128, 128], dtype=np.uint8),
         np.array([55, 128, 50], dtype=np.uint8)]

def add_mask2(image, mask, color_id):
    # 单通道
    int_mask = mask.astype(np.uint8)
    int_mask_3d = np.dstack((int_mask, int_mask, int_mask))

    mask_color = colors[color_id]
    mask = np.full_like(image, mask_color)

    mask[int_mask == 0] = 0

    # 使用 cv2.addWeighted 叠加原始图像和橙色掩码图像
    alpha = 0  # 原始图像权重
    beta = 1  # 橙色掩码权重
    gamma = 0  # 偏移量

    # 使用掩码矩阵来控制叠加
    res = cv2.addWeighted(image, alpha, mask, beta, gamma, dtype=cv2.CV_8U)

    # 将mask中为黑色部分保留原图,0的区域为True, 非零区域为False
    # 获取黑色区域
    black_areas = int_mask_3d == 0
    res[black_areas] = image[black_areas]
    return res


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Variables for annotation
points = {0: []}
labels = {0: []}
boxes = {0: []}
ann_obj_id = [0]
current_id = 0

# Variables for drawing boxes
start_point = None
end_point = None
is_drawing = False
drag_mode = False  # False: Point mode, True: Drag box mode


frame_names = sorted(
    [p for p in os.listdir(video_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))],
    key=lambda p: int(os.path.splitext(p)[0]),
)
frame_idx = 0
current_image = Image.open(os.path.join(video_dir, frame_names[frame_idx]))

def draw_interactions(ax):
    """
    Draw all interaction points and boxes on the current image.
    """
    ax.clear()
    ax.imshow(current_image)

    # Draw points
    for obj_id in points:
        for (x, y), label in zip(points[obj_id], labels[obj_id]):
            color = 'blue' if label == 1 else 'red'
            ax.plot(x, y, 'o', color=color, label=f"ID {obj_id}")

    # Draw boxes
    for obj_id in boxes:
        for box in boxes[obj_id]:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='green',
                facecolor='none',
            )
            ax.add_patch(rect)

    ax.set_title(f"frame {frame_idx}")
    ax.legend(loc="upper right")
    plt.draw()

def toggle_mode(event):
    """
    Toggle between point mode and box drag mode.
    """
    global drag_mode, is_drawing
    if is_drawing:  # Stop ongoing box drawing
        is_drawing = False
        print("Stopped box drawing due to mode switch.")

    # Toggle mode
    drag_mode = not drag_mode
    mode = "Box Drag Mode" if drag_mode else "Point Click Mode"
    print(f"Switched to {mode}")

    # Update button label
    btn.label.set_text(mode)
    plt.draw()

def increment_ann_id(event):
    """
    Increment and append the next integer to ann_obj_id, and initialize new interaction lists.
    """
    global points, labels, boxes, ann_obj_id, current_id  # Add current_id
    next_id = ann_obj_id[-1] + 1  # Increment ID
    ann_obj_id.append(next_id)
    current_id = next_id  # Update current ID

    # Initialize points, labels, and boxes for the new ID
    points[current_id] = []
    labels[current_id] = []
    boxes[current_id] = []

    print(f"Switched to ID: {current_id} | Current ann_obj_id: {ann_obj_id}")

    # Update the interactions on the current frame
    draw_interactions(ax)


def onclick(event):
    """
    Handle mouse clicks for points (only in point mode).
    """
    global is_drawing
    if event.inaxes == ax and not drag_mode and not is_drawing:  # Ensure in point mode
        x, y = int(event.xdata), int(event.ydata)
        current_id = ann_obj_id[-1]  # Get the current active ID

        if event.button == 1:  # Left click
            points[current_id].append([x, y])
            labels[current_id].append(1)
        elif event.button == 3:  # Right click
            points[current_id].append([x, y])
            labels[current_id].append(0)

        draw_interactions(ax)  # Update the display

def onpress(event):
    """
    Handle mouse press to start drawing a box (only in box drag mode).
    """
    global start_point, is_drawing
    if event.inaxes and drag_mode and event.button == 1:  # Only respond in drag mode
        start_point = (int(event.xdata), int(event.ydata))
        is_drawing = True

def onrelease(event):
    """
    Handle mouse release to finalize the box (only in box drag mode).
    """
    global start_point, end_point, is_drawing, boxes
    if is_drawing and event.inaxes and drag_mode and event.button == 1:  # Only respond in drag mode
        end_point = (int(event.xdata), int(event.ydata))
        x1, y1 = start_point
        x2, y2 = end_point
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]  # Ensure box ordering
        boxes[current_id].append(box)  # Append to the correct ID
        is_drawing = False
        draw_interactions(ax)  # Update display

# Load and display the image
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(bottom=0.2)  # Adjust layout to fit button
draw_interactions(ax)

# Add a button to toggle modes
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
btn = Button(ax_button, "Point Click Mode")
btn.on_clicked(toggle_mode)

ax_button_id = plt.axes([0.6, 0.05, 0.2, 0.075])  # Increment ID button
btn_id = Button(ax_button_id, "Increment ID")
btn_id.on_clicked(increment_ann_id)

# Connect the click and drag events
fig.canvas.mpl_connect('button_press_event', onpress)
fig.canvas.mpl_connect('button_release_event', onrelease)
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

from matplotlib.widgets import Button

def next_video(event):
    global current_video_idx, current_video_dir, frame_names, frame_idx
    current_video_idx = (current_video_idx + 1) % len(video_dirs)
    current_video_dir = video_dirs[current_video_idx]
    frame_names = sorted(
        [p for p in os.listdir(current_video_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    frame_idx = 0
    current_image = Image.open(os.path.join(current_video_dir, frame_names[frame_idx]))
    draw_interactions(ax)

def prev_video(event):
    global current_video_idx, current_video_dir, frame_names, frame_idx
    current_video_idx = (current_video_idx - 1) % len(video_dirs)
    current_video_dir = video_dirs[current_video_idx]
    frame_names = sorted(
        [p for p in os.listdir(current_video_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))],
        key=lambda p: int(os.path.splitext(p)[0]),
    )
    frame_idx = 0
    current_image = Image.open(os.path.join(current_video_dir, frame_names[frame_idx]))
    draw_interactions(ax)

def segment(event):
    global points, labels, boxes, current_id, inference_state, predictor
    ann_frame_idx = 0
    if boxes[current_id] is not None:
        boxes_obj = boxes[current_id][-1]
    else:
        boxes_obj = None
    points_obj = points[current_id]
    labels_obj = labels[current_id]
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=current_id,
            points=points_obj,
            labels=labels_obj,
            box=boxes_obj,
        )
        # Update the display with segmentation results
        draw_interactions(ax)
    except Exception as e:
        print(f"Segmentation failed: {e}")

# Add buttons for navigation and segmentation
ax_button_next = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_next = Button(ax_button_next, "Next Video")
btn_next.on_clicked(next_video)

ax_button_prev = plt.axes([0.5, 0.05, 0.1, 0.075])
btn_prev = Button(ax_button_prev, "Prev Video")
btn_prev.on_clicked(prev_video)

ax_button_segment = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_segment = Button(ax_button_segment, "Segment")
btn_segment.on_clicked(segment)



# Convert points and labels to numpy arrays
# Convert points and labels to numpy arrays, handle empty values correctly
if isinstance(points, dict):
    points = {k: (np.array(v, dtype=np.float32) if v else None) for k, v in points.items()}
else:
    points = np.array(points, dtype=np.float32) if points else None

if isinstance(labels, dict):
    labels = {k: (np.array(v, dtype=np.int32) if v else None) for k, v in labels.items()}
else:
    labels = np.array(labels, dtype=np.int32) if labels else None

if isinstance(boxes, dict):
    boxes = {k: (np.array(v, dtype=np.int32) if v else None) for k, v in boxes.items()}
else:
    boxes = np.array(boxes, dtype=np.int32) if boxes else None

# Debugging and further processing
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# ann_frame_idx and ann_obj_id can be used for further annotation logic
for i in ann_obj_id:
    
    ann_frame_idx = 0
    if boxes[i] != None:
        boxes_obj = boxes[i][-1]
    else:
        boxes_obj = None
    points_obj = points[i]
    labels_obj = labels[i]
    # inference_state = predictor.init_state(video_path=video_dir)
    # predictor.reset_state(inference_state)
    try:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=i,
            points=points_obj,
            labels=labels_obj,
            box=boxes_obj,
            )
    except:
        import pdb;pdb.set_trace()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation resultshjkly
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    print(out_obj_ids)
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


for out_frame_idx in range(0, len(frame_names)):
    color_num = 0
    now_img = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))
    now_img = np.full_like(now_img, 0)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        now_img = add_mask2(now_img, np.squeeze(out_mask),color_num)
        color_num += 1
    mask_out_name = '{}/{}.jpg'.format(output_dir, str(out_frame_idx).zfill(4))
    print(mask_out_name)
    cv2.imwrite(mask_out_name, now_img)

# save
# for out_num, out_mask in video_segments.items():


#     zero_matrix = np.zeros_like(next(iter(out_mask.values())), dtype=np.uint8)
#     # 创建一个用于存储结果的矩阵，初始为零矩阵
#     result_matrix = zero_matrix.copy()
    
#     # 遍历 out_mask 中的每个对象的掩码
#     for obj_id, mask in out_mask.items():
#         # 将布尔掩码转换为 8 位整数掩码
#         bw_image = mask.astype(np.uint8)
#         print(f"对象 ID: {obj_id}, 二值图像的形状: {bw_image.shape}")
#         # 将当前掩码与结果矩阵进行按位或操作
#         result_matrix |= bw_image
#     # Create an image from the array
#     img = Image.fromarray(result_matrix.squeeze()*255)
#     out_obj_id = int(out_num)
#     # Construct the filename
#     filename = f"{str(out_obj_id).zfill(4)}.png"
    
#     # Save the image
#     img.save(os.path.join(output_dir, filename))

# for frame_idx, frame_data in video_segments.items():
#     # 加载原始图像
#     original_image = np.array(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

#     for out_obj_id, mask in frame_data.items():
#         # 将 mask 去掉第一个维度，变为 (100, 100)
#         mask = mask.squeeze(0)  # 从 (1, 100, 100) 转为 (100, 100)

#         # 将 mask 扩展为 (100, 100, 3)，以便与原图对齐
#         mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

#         # 创建一个与原图相同的空白图片
#         filtered_image = np.zeros_like(original_image, dtype=np.uint8)

#         # 将原图中 mask 为 True 的像素保留下来
#         filtered_image[mask > 0] = original_image[mask > 0]

#         # 保存结果图像
#         filtered_image = Image.fromarray(filtered_image)
#         filtered_filename = f"{str(frame_idx).zfill(4)}.jpg"
#         filtered_image.save(os.path.join(output_filtered_dir, filtered_filename))

# print(f"Filtered images saved to {output_filtered_dir}")
