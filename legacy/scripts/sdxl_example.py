import torch

from legacy.pipefuser.pipelines import DistriSDXLPipeline
from legacy.pipefuser.utils import DistriConfig


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
#        default="stabilityai/stable-diffusion-xl-base-1.0",
        default="/home/jovyan/models/stable-diffusion-xl-base-1.0/models/snapshots/462165984030d82259a11f4367a4eed129e94a7b",
        type=str,
        help="Path or Id to the pretrained model.",
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
    args = parser.parse_args()

    distri_config = DistriConfig(height=args.height, width=args.width, warmup_steps=1)

    pipeline = DistriSDXLPipeline.from_pretrained(
        distri_config=distri_config,
        pretrained_model_name_or_path=args.model_id,
        variant="fp16",
        use_safetensors=True,
    )

    pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    image = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    import time
    t1=time.time()
    image = pipeline(
        prompt="An astronaut riding a green horse",
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    t2=time.time()
    print("sdxl execute time ", t2-t1)
    if distri_config.rank == 0:
        image.save("astronaut.png")
