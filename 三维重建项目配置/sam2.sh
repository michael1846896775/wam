#!/bin/bash
source /home/wangam/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate sam2
cd /home/wangam/segmention/sam2
python annotate.py --root_dir "$1" --annotation_file "$1/all_annotations.txt"
python segment.py --annotation_file "$1/all_annotations.txt"