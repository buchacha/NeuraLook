import os
import io
import warnings
import argparse
import time
import pathlib

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import numpy as np


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Stable Diffusion")

    parser.add_argument(
        "--input-dir", type=str, default="", help="path of input image folder."
    )
    parser.add_argument(
        "--mask-dir", type=str, default="", help="path of mask of image folder."
    )
    parser.add_argument(
        "--cont", type=str, default="", help="path to collect processed images."
    )
    parser.add_argument(
        "--search", type=str, default="", help="path of images using for search."
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="path of output image folder."
    )

    return parser.parse_args()


def setup_env():
    os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"
    os.environ["STABILITY_KEY"] = "sk-wJBsRNL0QF0ET6ac08Yim4YjIXBFwP6jPsko5pMmVqxmIQTX"

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ["STABILITY_KEY"],  # API Key reference.
        verbose=True,  # Print debug messages.
        engine="stable-inpainting-v1-0",  # Set the engine to use for generation.
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    return stability_api


def create_box(mask):
    ar = np.array(mask)
    morphed_white_pixels = np.argwhere(ar == 0)

    min_y = min(morphed_white_pixels[:, 1])
    max_y = max(morphed_white_pixels[:, 1])
    min_x = min(morphed_white_pixels[:, 0])
    max_x = max(morphed_white_pixels[:, 0])

    padding = 20
    top_left = (min_y - padding, min_x - padding)
    bottom_right = (max_y + padding, max_x + padding)
    return top_left, bottom_right


def get_promt(image_path):
    base_prompt = []
    opt_additional_prompts = "photorealistic, best quality"
    opt_additional_prompts_weight = 1
    opt_antiprompts = "face, cartoon, nude, the clothes fit perfectly, tattoo"
    opt_antiprompts_weight = -1
    basename = os.path.basename(image_path)
    list_from_basename = basename[:-4].split(sep="_")
    prompt_list = list_from_basename[1:-1]
    prompt = " ".join(prompt_list)
    base_prompt = [
        generation.Prompt(
            text=f"{prompt}, {opt_additional_prompts}",
            parameters=generation.PromptParameters(
                weight=opt_additional_prompts_weight
            ),
        )
    ]
    base_prompt.append(
        generation.Prompt(
            text=f"{opt_antiprompts}",
            parameters=generation.PromptParameters(weight=opt_antiprompts_weight),
        )
    )
    return base_prompt


def create_request(api, prompt, image, mask):
    opt_sh = 0.8  # Степень влияния исходной картинки на начальных шагах (0-1) (меньше значение - больше влияние)
    opt_end_sh = 0.1  # Степень влияния исходной картинки на конечных шагах (0-1) (меньше значение - больше влияние)
    opt_cfg = 7  # (1-35) Насколько сильно модель следует промту
    opt_steps = 30  # Количество шагов генерации (10-150)
    opt_sampler = generation.SAMPLER_K_EULER_ANCESTRAL

    answers = api.generate(
        prompt=prompt,
        init_image=image,
        mask_image=mask,
        # mask_source='MASK_IMAGE_BLACK',
        start_schedule=opt_sh,
        end_schedule=opt_end_sh,
        samples=4,
        # If attempting to transform an image that was previously generated with our API,
        # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
        steps=opt_steps,  # Amount of inference steps performed on image generation. Defaults to 30.
        cfg_scale=opt_cfg,  # Influences how strongly your generation is guided to match your prompt.
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=mask.size[0],  # Generation width, defaults to 512 if not included.
        height=mask.size[1],  # Generation height, defaults to 512 if not included.
        sampler=opt_sampler  # Choose which sampler we want to denoise our generation with.
        # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )
    return answers


def save_cropped(img, img_name, search_path, top_left, bottom_right):
    res_im = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    res_im.save(os.path.join(search_path, img_name))


def save_answer(answers, img_name, save_path, search_path, left_top, right_bottom):
    i = 0
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again."
                )
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(os.path.join(save_path, img_name[:-4] + f"_{i}.png"))
                save_cropped(
                    img,
                    img_name[:-4] + f"_{i}.png",
                    search_path,
                    left_top,
                    right_bottom,
                )
                i = i + 1


def main():
    args = get_arguments()
    print("a")

    current_path = pathlib.Path(__file__).parent.resolve()
    args.input_dir = os.path.join(current_path, "container_segmentation")
    args.container = os.path.join(current_path, "container_generation")
    args.mask_dir = os.path.join(current_path, "output_segmentation")

    if not os.path.exists(args.container):
        os.makedirs(args.container)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.search):
        os.makedirs(args.search)
    print("a")
    while True:
        api = setup_env()
        list_dir = os.listdir(args.input_dir)
        print("a")
        if list_dir == []:
            time.sleep(10)
        else:
            for img_path in list_dir:
                prompt = get_promt(img_path)
                image = Image.open(os.path.join(args.input_dir, img_path))
                mask = Image.open(os.path.join(args.mask_dir, img_path)).convert("L")
                left_top, right_bottom = create_box(mask)
                answers = create_request(api, prompt, image, mask)
                try:
                    os.rename(
                        os.path.join(args.input_dir, img_path),
                        os.path.join(args.cont, img_path),
                    )
                except FileExistsError:
                    os.remove(os.path.join(args.input_dir, img_path))
                save_answer(
                    answers,
                    img_path,
                    args.output_dir,
                    args.search,
                    left_top,
                    right_bottom,
                )
                if img_path[:-4] == "stop_the_process":
                    return


if __name__ == "__main__":
    main()
