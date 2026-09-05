#!/bin/bash
source /home/wangam/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate gaussian_splatting
cd /home/wangam/gaussian-splatting
python pitrain.py --main_dir "$1"