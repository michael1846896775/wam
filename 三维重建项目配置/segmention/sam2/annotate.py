import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button

# 创建解析器
parser = argparse.ArgumentParser(description="Batch interactive annotation tool for SAM2")

# 添加命令行参数
parser.add_argument(
    '--root_dir', 
    type=str, 
    default="/home/wangam/data_demo", 
    help="Root directory containing video folders"
)

parser.add_argument(
    '--annotation_file', 
    type=str, 
    default='./all_annotations.txt', 
    help="File to save all annotation data"
)

# 解析命令行参数
args = parser.parse_args()

# 获取传入的参数
root_dir = args.root_dir
annotation_file = args.annotation_file

print(f"Root directory: {root_dir}")
print(f"Annotation file: {annotation_file}")

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

def find_video_folders(root_dir):
    """
    查找所有包含images文件夹的视频文件夹
    """
    video_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            images_path = os.path.join(item_path, "images")
            if os.path.exists(images_path) and os.path.isdir(images_path):
                video_folders.append(item_path)
                print(f"Found video folder: {item_path}")
    
    return sorted(video_folders)

class AnnotationState:
    """管理标注状态的类"""
    def __init__(self):
        self.points = {0: []}
        self.labels = {0: []}
        self.boxes = {0: []}
        self.ann_obj_id = [0]
        self.current_id = 0
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.drag_mode = False

def process_single_video(video_dir):
    """
    处理单个视频文件夹的交互式标注
    """
    # 创建状态对象
    state = AnnotationState()

    # 查找图像文件
    images_dir = os.path.join(video_dir, "images")
    frame_names = sorted(
        [p for p in os.listdir(images_dir) if p.endswith((".jpg", ".jpeg", ".JPG", ".JPEG", ".png"))],
        key=lambda p: int(os.path.splitext(p)[0]) if p.split('.')[0].isdigit() else p,
    )
    
    if not frame_names:
        print(f"No image files found in {images_dir}")
        return None
        
    frame_idx = 0
    current_image = Image.open(os.path.join(images_dir, frame_names[frame_idx]))

    def draw_interactions(ax):
        ax.clear()
        ax.imshow(current_image)

        # Draw points
        for obj_id in state.points:
            for (x, y), label in zip(state.points[obj_id], state.labels[obj_id]):
                color = 'blue' if label == 1 else 'red'
                ax.plot(x, y, 'o', color=color, label=f"ID {obj_id}")

        # Draw boxes
        for obj_id in state.boxes:
            for box in state.boxes[obj_id]:
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

        ax.set_title(f"{os.path.basename(video_dir)} - frame {frame_idx}")
        ax.legend(loc="upper right")
        plt.draw()

    def toggle_mode(event):
        if state.is_drawing:
            state.is_drawing = False
            print("Stopped box drawing due to mode switch.")

        state.drag_mode = not state.drag_mode
        mode = "Box Drag Mode" if state.drag_mode else "Point Click Mode"
        print(f"Switched to {mode}")

        btn.label.set_text(mode)
        plt.draw()

    def increment_ann_id(event):
        next_id = state.ann_obj_id[-1] + 1
        state.ann_obj_id.append(next_id)
        state.current_id = next_id

        state.points[state.current_id] = []
        state.labels[state.current_id] = []
        state.boxes[state.current_id] = []

        print(f"Switched to ID: {state.current_id} | Current ann_obj_id: {state.ann_obj_id}")
        draw_interactions(ax)

    def onclick(event):
        if event.inaxes == ax and not state.drag_mode and not state.is_drawing:
            x, y = int(event.xdata), int(event.ydata)
            current_id = state.ann_obj_id[-1]

            if event.button == 1:  # Left click
                state.points[current_id].append([x, y])
                state.labels[current_id].append(1)
            elif event.button == 3:  # Right click
                state.points[current_id].append([x, y])
                state.labels[current_id].append(0)

            draw_interactions(ax)

    def onpress(event):
        if event.inaxes and state.drag_mode and event.button == 1:
            state.start_point = (int(event.xdata), int(event.ydata))
            state.is_drawing = True

    def onrelease(event):
        if state.is_drawing and event.inaxes and state.drag_mode and event.button == 1:
            state.end_point = (int(event.xdata), int(event.ydata))
            x1, y1 = state.start_point
            x2, y2 = state.end_point
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            state.boxes[state.current_id].append(box)
            state.is_drawing = False
            draw_interactions(ax)

    # 创建交互界面
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.15)
    draw_interactions(ax)

    # 添加按钮
    ax_button = plt.axes([0.3, 0.05, 0.2, 0.06])
    btn = Button(ax_button, "Point Click Mode")
    btn.on_clicked(toggle_mode)

    ax_button_id = plt.axes([0.5, 0.05, 0.2, 0.06])
    btn_id = Button(ax_button_id, "Increment ID")
    btn_id.on_clicked(increment_ann_id)

    # 连接事件
    fig.canvas.mpl_connect('button_press_event', onpress)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.suptitle(f"Processing: {os.path.basename(video_dir)}", fontsize=14)
    plt.show()
    
    # 返回当前视频的标注数据
    annotation_data = {
        'video_dir': video_dir,
        'points': {str(k): v for k, v in state.points.items()},
        'labels': {str(k): v for k, v in state.labels.items()},
        'boxes': {str(k): v for k, v in state.boxes.items()},
        'ann_obj_id': state.ann_obj_id,
        'frame_idx': frame_idx
    }
    
    return annotation_data

def save_all_annotations(annotation_data_list):
    """
    保存所有视频的标注数据到一个文件
    """
    os.makedirs(os.path.dirname(annotation_file) if os.path.dirname(annotation_file) else '.', exist_ok=True)
    
    all_annotations = {
        'videos': annotation_data_list
    }
    
    try:
        with open(annotation_file, 'w') as f:
            json.dump(all_annotations, f, indent=4)
        print(f"All annotations saved to {annotation_file}")
    except Exception as e:
        print(f"Error saving annotations to {annotation_file}: {e}")

def main():
    # 查找所有视频文件夹
    video_folders = find_video_folders(root_dir)
    
    if not video_folders:
        print(f"No video folders found in {root_dir}")
        return
    
    print(f"Found {len(video_folders)} video folders to process")
    
    all_annotations = []
    
    for i, video_folder in enumerate(video_folders, 1):
        folder_name = os.path.basename(video_folder)
        
        print(f"\n{'='*60}")
        print(f"Processing folder {i}/{len(video_folders)}: {folder_name}")
        print(f"{'='*60}")
        
        # 处理当前视频文件夹
        annotation_data = process_single_video(video_folder)
        
        if annotation_data:
            all_annotations.append(annotation_data)
            print(f"Completed processing: {folder_name}")
        else:
            print(f"Failed to process: {folder_name}")
        
        print(f"Progress: {i}/{len(video_folders)} completed")
    
    # 保存所有标注数据到一个文件
    save_all_annotations(all_annotations)
    print(f"\nBatch processing completed! Processed {len(all_annotations)} folders.")
    print(f"All annotations saved to: {annotation_file}")

if __name__ == "__main__":
    main()