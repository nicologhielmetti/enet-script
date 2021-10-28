#!/bin/bash

while getopts ":r:f:c:q:p:i:o:" opt; do
  case $opt in
    r) reuse_factor="$OPTARG"
    ;;
    f) n_filters="$OPTARG"
    ;;
    c) clock_period="$OPTARG"
    ;;
    q) quantization="$OPTARG"
    ;;
    p) precision="$OPTARG"
    ;;
    i) input_data="$OPTARG"
    ;;
    o) output_predictions="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

source /home/hls4ml/.bashrc
conda activate hls4ml-testing
pip install git+https://github.com/nicologhielmetti/hls4ml.git@fifo_depth_opt#egg=hls4ml[profiling]
pip install keras
source /opt/Xilinx/Vivado/2019.2/settings64.sh
python explore-enet.py -r $reuse_factor -f $n_filters -c $clock_period -q $quantization -p $precision -i $input_data -o $output_predictions
