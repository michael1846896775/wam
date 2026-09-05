#!/bin/bash
source /home/wangam/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate vggsfm_tmp
cd /home/wangam/vggsfm
python pidemo.py --base_dir "$1"