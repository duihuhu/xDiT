#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_batch_size.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt "black sky, all stars" --height 512 --width 512
sleep 5

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_batch_size.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt ["black sky, all stars", "black sky, all stars"] --height 512 --width 512
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_batch_size.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt ["black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars"] --height 512 --width 512
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_batch_size.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt ["black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars"] --height 512 --width 512
sleep 5


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 pixartalpha_example_batch_size.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt ["black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars", "black sky, all stars"] --height 512 --width 512
sleep 5
