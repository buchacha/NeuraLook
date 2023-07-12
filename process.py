#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset


def find_new_size(side_ratio):
    """
    Функция для нахождения максимального размера картинки с заданным соотношением 
    сторон. В качестве входного элемента `side_ratio` принимает соотношение сторон
    и вычисляет максимальный возможный размер изображения, используя ограничение на 
    общее количество пикселей в Stable Diffusion (1024*1024=1048576) и ограничение 
    на размер изображения: высота и ширина должны делиться на 64
    Parameters:
    side_ratio - bigger side size in pixels / smaller side size in pixels
    """
    if side_ratio > 1:
        return int((1048576 / side_ratio)**0.5)//64*64, int(((1048576 / side_ratio)**0.5)*side_ratio)//64*64
    else:
        return int(((1048576 / side_ratio)**0.5)*side_ratio)//64*64, int((1048576 / side_ratio)**0.5)//64*64


def remove_head_from_mask(mask, head_mask):
    """
    Функция для удаления пикселей лица с маски. В качестве входных аргументов 
    принимает 2 маски: `mask` - размытая маска рук и верхней одежды и `head_mask` -
    маска головы. Возвращает изображение маски со стёртыми пикселями лица
    Parameters:
    mask - dilated mask
    head_mask - mask of head
    """
    mask_ar = np.array(mask)
    head_mask_ar = np.array(head_mask)
    mask_without_head_ar = mask_ar - head_mask_ar
    mask_without_head_ar = np.where(mask_without_head_ar > 1, 0, 255)
    mask_without_head_ar = np.asarray(mask_without_head_ar, dtype=np.uint8)
    mask_without_head = Image.fromarray(mask_without_head_ar, mode='RGB')
    return mask_without_head
    

def preproc_img_size(img):
    """
    Функция для приведения изображения к нужному размеру: либо максимально возможному
    с исходным соотношением сторон,если исходное изображение содержит больше 2**20 пикселей, 
    либо к размеру, где ширина и высота изображения делятся на 64, если размер изображения
    не превышает максимально возможный для Stable Diffusion. Возвращает изображение с 
    "правильным" размером
    Parameters:
    img - image to resize 
    """
    min_size_index = np.argmin(img.size)
    side_ratio = img.size[min_size_index] / img.size[min_size_index-1]
    new_size = find_new_size(side_ratio)
    m = [0, 0]
    if img.size[min_size_index] > new_size[0]:
        m[min_size_index] = new_size[0]
        m[min_size_index-1] = new_size[1]
        return img.resize((m))
    else:
        m[0] = img.size[0]//64*64
        m[1] = img.size[1]//64*64
        return img.resize((m))
   


def dilation(img, iterations):
    """
    Функция для размытия маски изображения. Принимает на вход `img` - исходная маска, и 
    `iteration` - количество итераций размытия, чем больше - тем сильнее размытие
    """
    img2 = np.array(img)
    img2 = img2[:, :, ::-1].copy() 
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(img2,kernel,iterations = iterations)
    dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
    return Image.fromarray(dilation)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--cont", type=str, default='', help="path to collect processed images")

    return parser.parse_args()


def main():
    args = get_arguments() # собираем входные аргументы
    # Настраиваем окружение и дефолтные параметры
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    num_classes = 18
    input_size = [512, 512]
    # Загружаем модель
    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict'] 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # определяем палетты для масок головы и основной маски
    head_palette = [item for sublist in [[0]*11*3, [255]*3, [0]*6*3] for item in sublist]
    palette = [item for sublist in [[0]*12, [255]*3, [0]*9*3, [255]*6, [0]*6] for item in sublist]
    while True: # начинаем постоянный цикл
        if os.listdir(args.input_dir) == []:
            time.sleep(10)  # если во входной папке ничего нет - засываем
        else:
            dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)  # если есть - собираем датасет из входной папки
            dataloader = DataLoader(dataset)
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader)):  # перемещаемся по датасету
                    # определяем параметры изображения
                    image, meta = batch
                    img_name = meta['name'][0]
                    c = meta['center'].numpy()[0]
                    s = meta['scale'].numpy()[0]
                    w = meta['width'].numpy()[0]
                    h = meta['height'].numpy()[0]

                    # получаем результаты работы сегментатора
                    output = model(image.cuda())
                    upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                    upsample_output = upsample_output.squeeze()
                    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                    # преобразуем результаты в правильный вид
                    logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                    parsing_result = np.argmax(logits_result, axis=2)

                    # создаём пути для сохранения изображений
                    parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
                    save_to_container_path = os.path.join(args.cont, img_name[:-4] + '.png')

                    # обрабатываем выходное изображение
                    cont_image = preproc_img_size(Image.open(os.path.join(args.input_dir, img_name)))   # обрабатываем размер входного изображения
                    cont_image.save(save_to_container_path) # сохраняем входное изображение с изменённым размером (тоже пригодится для Stable Diffusion)
                    os.remove(os.path.join(args.input_dir, img_name))   # удаляем входное изображение
                    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))    # собираем выходное изображение
                    head_img = output_img.copy()    # сразу копируем его для последующего создания маски головы
                    head_img.putpalette(head_palette)   # применяем палетту для создания маски головы
                    output_img.putpalette(palette)  # применяем палетту для создания маски верхней одежды и рук
                    dil_param = img_name[:-4].split(sep='_')[-1]    # из названия файла выделяем число (предполагается формат названия файла типа id_promp_dilparam.png)
                    output_img = dilation(output_img.convert('RGB'), int(dil_param))    # размываем маску верхней одежды и рук
                    output_img = remove_head_from_mask(output_img.convert('RGB'), head_img.convert('RGB'))  # убираем из маски голову
                    output_img = preproc_img_size(output_img)   # обрабатываем размер изображения
                    output_img.save(parsing_result_path)    # сохраняем маску
                    if img_name[:-4] == 'stop_the_process': # способ закончить цикл while True)
                        return
                    
                    


if __name__ == '__main__':
    main()
