#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 64 --width 64
sleep 5

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 128 --width 128
sleep 5

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 256 --width 256
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 512 --width 512
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 1024 --width 1024
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 2048 --width 2048
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 4096 --width 4096
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_repreated.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 20 --warmup_steps 0 --prompt "black sky, all stars" --height 8192 --width 8192
sleep 5