import os
import io
import warnings
import argparse
import time

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import numpy as np
import config   # здесь хранится токен для Stable Diffusion


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Stable Diffusion API request")

    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--mask-dir", type=str, default='', help="path of mask of image folder.")
    parser.add_argument("--cont", type=str, default='', help="path to collect processed images.")
    parser.add_argument("--search", type=str, default='', help="path of images using for search.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")

    return parser.parse_args()


def setup_env():
    """
    Функция для подключения к Stable Diffusion API
    """
    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
    os.environ['STABILITY_KEY'] = config.token

    # Set up our connection to the API.
    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'], # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-768-v2-1", # Set the engine to use for generation. 
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    return stability_api


def create_box(mask):
    """
    Функция для создания квадрата для сохранения изображений, которые будут использоваться при поиске товара.
    Принимает на вход `mask` и возвращает левый верхний и правый нижний пиксель, по которым далее надо будет 
    обрезать сгенерированное изображение, чтобы получить изображение для поискового запроса
    """
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


def get_prompt(image_path):
    """"
    Функция для создания промпта при помощи названия файла. Предполагается, что формат названия 
    следующий: "id_{prompt}_dilparam.png", где id - идентификатор пользователя или что-то подобное,  
    {prompt} - промпт, а dilparam - параметр для функции dilation из process.py. 
    Пример: 192133123_men_orange_jacket_5.png
    Возвращает промпт с заранее заданными положительными и отрицательными параметры и промптом из
    названия изображения
    """
    base_prompt = []
    basename = os.path.basename(image_path)
    list_from_basename = basename[:-4].split(sep='_')
    prompt_list = list_from_basename[1:-1]
    prompt = ' '.join(prompt_list)
    base_prompt = [generation.Prompt(text=f'{prompt}, photorealistic, best quality', parameters=generation.PromptParameters(weight=1))]
    base_prompt.append(generation.Prompt(text='face, cartoon, nude',parameters=generation.PromptParameters(weight=-1)))
    return base_prompt


def create_request(api, prompt, image, mask):
    """
    Функция, создающая запрос к API по заданным параметрам
    Наиболее интересные из дефолтных параметры для изменения результатов генерации - `start_schedule` 
    и `cfg_scale`. Подробнее о них: https://platform.stability.ai/docs/features/api-parameters
    """
    answers = api.generate(
    prompt=prompt,
    init_image=image,
    mask_image=mask,
    # mask_source='MASK_IMAGE_BLACK',
    start_schedule=1,
    samples=4,
     # If attempting to transform an image that was previously generated with our API,
                    # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
    steps=25, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=9.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=mask.size[0], # Generation width, defaults to 512 if not included.
    height=mask.size[1], # Generation height, defaults to 512 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )   
    return answers


def save_cropped(img, img_name, search_path, top_left, bottom_right):
    """
    Функция для сохранения обрезанных изображений для поиска
    """
    res_im = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
    res_im.save(os.path.join(search_path, img_name))


def save_answer(answers, img_name, save_path, search_path, left_top, right_bottom):
    """
    Функция для сохранения изображений, полученных от API
    """
    i = 0
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                global img
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(os.path.join(save_path, img_name[:-4]+ f"_{i}.png"))
                save_cropped(img, img_name[:-4]+ f"_{i}.png", search_path, left_top, right_bottom) # здесь же сразу сохраняются изображения для поиска
                i = i + 1




def main():
    args = get_arguments()  # собираем аргументы
    while True:
        api = setup_env()   # подключение к API в цикле, т.к. иногда бывают сбои и сбросы соединения
        list_dir = os.listdir(args.input_dir)   # сканируем входную папку на наличие файлов
        if list_dir == []:
            time.sleep(10)  # если ничего нет, то спим
        else:               # если есть, то работаем
            for img_path in list_dir:
                prompt = get_prompt(img_path)   # получаем промпт из названия
                image = Image.open(os.path.join(args.input_dir, img_path))  # открываем изображение
                mask = Image.open(os.path.join(args.mask_dir, img_path)).convert('L')   # открываем маску
                left_top, right_bottom = create_box(mask)   # создаём координаты для обрезания выходных изображений
                answers = create_request(api, prompt, image, mask)  # делаем запрос к API
                try:
                    os.rename(os.path.join(args.input_dir, img_path), os.path.join(args.cont, img_path))    # перемещаем обработанные изображения из папки
                except FileExistsError:
                    os.remove(os.path.join(args.input_dir, img_path))   # если уже существуют такие, то удаляем
                save_answer(answers, img_path, args.output_dir, args.search, left_top, right_bottom)    # сохраняем то, что получилось
                if img_path[:-4] == 'stop_the_process':     # снова способ выхода из цикла while True
                    return
                

if __name__ == '__main__':
    main()




                

