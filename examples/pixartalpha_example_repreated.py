#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=1 examples/pixartalpha_example.py --model /home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/  --num_inference_steps 40 --warmup_steps 0 --prompt "black sky, all stars" --height 512 --width 512
import time
import os
import torch
import torch.distributed
from xfuser import xFuserPixArtAlphaPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.distributed import (
    get_world_group,
    is_dp_last_rank,
    get_data_parallel_world_size,
    get_runtime_state
)


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank
    pipe = xFuserPixArtAlphaPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    pipe.prepare_run(input_config)
    torch.cuda.reset_peak_memory_stats()
    output = pipe(
        height=input_config.height,
        width=input_config.height,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        use_resolution_binning=input_config.use_resolution_binning,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    sum_time = 0
    for i in range(5):
        start_time = time.time()
        output = pipe(
            height=input_config.height,
            width=input_config.height,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            use_resolution_binning=input_config.use_resolution_binning,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )
        torch.cuda.synchronize() 
        end_time = time.time()
        elapsed_time = end_time - start_time
        sum_time = sum_time + elapsed_time

    print("avg_time ", sum_time/5)
    print("avg step time ", sum_time/(5*input_config.num_inference_steps))
       
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if input_config.output_type == "pil":
        global_rank = get_world_group().rank
        dp_group_world_size = get_data_parallel_world_size()
        dp_group_index = global_rank // dp_group_world_size
        num_dp_groups = engine_config.parallel_config.dp_degree
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if is_dp_last_rank():
            if not os.path.exists('results'):
                os.mkdir('results')
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image.save(f"./results/pixart_alpha_result_{parallel_info}_{image_rank}_{i}.png")


    get_runtime_state().destory_distributed_env()


if __name__ == '__main__':
    main()