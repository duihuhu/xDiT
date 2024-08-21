import argparse
import torch
import torch.distributed as dist

from legacy.pipefuser.pipelines.pixartalpha import DistriPixArtAlphaPipeline
from legacy.pipefuser.utils import DistriConfig
from torch.profiler import profile, ProfilerActivity

import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="/home/jovyan/models/PixArt-XL-2-1024-MS/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404/",
        type=str,
        help="Path to the pretrained model.",
    )
    parser.add_argument(
        "--parallelism",
        "-p",
        default="patch",
        type=str,
        choices=["patch", "naive_patch", "pipefusion", "tensor", "sequence"],
        help="Parallelism to use.",
    )
    parser.add_argument(
        "--sync_mode",
        type=str,
        default="corrected_async_gn",
        choices=[
            "separate_gn",
            "async_gn",
            "corrected_async_gn",
            "sync_gn",
            "full_sync",
            "no_sync",
        ],
        help="Different GroupNorm synchronization modes",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--pp_num_patch", type=int, default=2, help="patch number in pipefusion."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of image",
    )
    parser.add_argument(
        "--no_use_resolution_binning",
        action="store_true",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pipefusion_warmup_step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_use_ulysses_low",
        action="store_true",
    )
    # parser.add_argument(
    #     "--use_profiler",
    #     action="store_true",
    # )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
    )
    parser.add_argument(
        "--use_parallel_vae",
        action="store_true",
    )
    parser.add_argument(
        "--use_split_batch",
        action="store_true",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="latent",
        choices=["latent", "pil"],
        help="latent saves memory, pil will results a memory burst in vae",
    )
    parser.add_argument("--attn_num", default=None, nargs="*", type=int)
    parser.add_argument(
        "--scheduler",
        "-s",
        default="dpm-solver",
        type=str,
        choices=["dpm-solver", "ddim"],
        help="Scheduler to use.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="An astronaut riding a green horse",
    )
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True

    enable_parallel_vae = args.use_parallel_vae

    # for DiT the height and width are fixed according to the model
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        warmup_steps=args.pipefusion_warmup_step,
        do_classifier_free_guidance=True,
        split_batch=args.use_split_batch,
        parallelism=args.parallelism,
        mode=args.sync_mode,
        pp_num_patch=args.pp_num_patch,
        use_resolution_binning=not args.no_use_resolution_binning,
        use_cuda_graph=args.use_cuda_graph,
        attn_num=args.attn_num,
        scheduler=args.scheduler,
        ulysses_degree=args.ulysses_degree,
    )
    torch.distributed.barrier()

    pipeline = DistriPixArtAlphaPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        enable_parallel_vae=enable_parallel_vae,
        # use_profiler=args.use_profiler,
    )

    rank = dist.get_rank()
    pipeline.set_progress_bar_config(disable=rank != 0)
    torch.distributed.barrier()
    # warmup
    output = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(42),
        output_type=args.output_type,
        num_inference_steps=args.pipefusion_warmup_step + 1,
    )

    torch.cuda.reset_peak_memory_stats()

    if args.parallelism == "pipefusion":
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_u{args.ulysses_degree}_w{distri_config.world_size}_mb{args.pp_num_patch}_warm{args.pipefusion_warmup_step}"
    else:
        case_name = f"{args.parallelism}_hw_{args.height}_sync_{args.sync_mode}_u{args.ulysses_degree}_w{distri_config.world_size}"
    if args.output_file:
        case_name = args.output_file + "_" + case_name
    if enable_parallel_vae:
        case_name += "_patchvae"
    if args.use_split_batch:
        case_name += "_split_batch"

    # if args.use_profiler:
    #     start_time = time.time()
    #     with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #             f"./profile/{case_name}"
    #         ),
    #         profile_memory=True,
    #         with_stack=True,
    #         record_shapes=True,
    #     ) as prof:
    #         output = pipeline(
    #             prompt=args.prompt,
    #             generator=torch.Generator(device="cuda").manual_seed(42),
    #             num_inference_steps=args.num_inference_steps,
    #             output_type=args.output_type,
    #         )
    #     # if distri_config.rank == 0:
    #     #     prof.export_memory_timeline(
    #     #         f"{distri_config.mode}_{args.height}_{distri_config.world_size}_mem.html"
    #     #     )
    #     end_time = time.time()
    # else:
        # MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
        # torch.cuda.memory._record_memory_history(
        #     max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        # )
    start_time = time.time()
    output = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(42),
        num_inference_steps=args.num_inference_steps,
        output_type=args.output_type,
    )

    end_time = time.time()
    start_time = time.time()
    output = pipeline(
        prompt=args.prompt,
        generator=torch.Generator(device="cuda").manual_seed(42),
        num_inference_steps=args.num_inference_steps,
        output_type=args.output_type,
    )

    end_time = time.time()
    print("pixart execute time ", end_time-start_time)
    # torch.cuda.memory._dump_snapshot(
    #     f"{distri_config.mode}_{distri_config.world_size}.pickle"
    # )
    torch.cuda.memory._record_memory_history(enabled=None)

    elapsed_time = end_time - start_time

    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    dist.barrier()

    # if rank == 0:
    # if distri_config.rank == 0:
    if dist.get_rank() == 0:

        print(
            f"{case_name} epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB"
        )
        if args.output_type == "pil":
            print(f"save images to ./results/{case_name}.png")
            output.images[0].save(f"./results/{case_name}.png")


if __name__ == "__main__":
    main()
