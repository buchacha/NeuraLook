#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import cv2
import torch
import argparse
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset


def find_new_size(side_ratio):
    if side_ratio > 1:
        return (
            int((1048576 / side_ratio) ** 0.5) // 64 * 64,
            int(((1048576 / side_ratio) ** 0.5) * side_ratio) // 64 * 64,
        )
    else:
        return (
            int(((1048576 / side_ratio) ** 0.5) * side_ratio) // 64 * 64,
            int((1048576 / side_ratio) ** 0.5) // 64 * 64,
        )


def remove_head_from_mask(mask, head_mask):
    mask_ar = np.array(mask)
    head_mask_ar = np.array(head_mask)
    mask_without_head_ar = mask_ar - head_mask_ar
    mask_without_head_ar = np.where(mask_without_head_ar > 1, 0, 255)
    mask_without_head_ar = np.asarray(mask_without_head_ar, dtype=np.uint8)
    mask_without_head = Image.fromarray(mask_without_head_ar, mode="RGB")
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
    side_ratio = img.size[min_size_index] / img.size[min_size_index - 1]
    new_size = find_new_size(side_ratio)
    m = [0, 0]
    if img.size[min_size_index] > new_size[0]:
        m[min_size_index] = new_size[0]
        m[min_size_index - 1] = new_size[1]
        return img.resize((m))
    else:
        m[0] = img.size[0] // 64 * 64
        m[1] = img.size[1] // 64 * 64
        return img.resize((m))


def erosion(img, iterations, dilation_kernel_size=(5, 5)):
    """
    Функция для обратной эрозии маски изображения. Принимает на вход `img` - исходная маска, и
    `iteration` - количество итераций эрозии, чем больше - тем сильнее эффект
    """
    img2 = np.array(img)
    img2 = img2[:, :, ::-1].copy()
    kernel = np.ones(dilation_kernel_size, np.uint8)
    dilation = cv2.erode(img2, kernel, iterations=iterations)
    dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
    return Image.fromarray(dilation)


def blur(img, blur_kernel_size, sigma):
    """
    Функция для размытия маски изображения, чем больше сигма - тем сильнее
    """
    blur = transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=sigma)
    return blur(img)


def dilation(img, iterations):
    img2 = np.array(img)
    img2 = img2[:, :, ::-1].copy()
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img2, kernel, iterations=iterations)
    dilation = cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB)
    return Image.fromarray(dilation)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument(
        "--model-restore",
        type=str,
        default="",
        help="restore pretrained model parameters.",
    )
    parser.add_argument("--gpu", type=str, default="0", help="choose gpu device.")
    parser.add_argument(
        "--input-dir", type=str, default="", help="path of input image folder."
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="path of output image folder."
    )
    parser.add_argument(
        "--container", type=str, default="", help="path to collect processed images"
    )

    return parser.parse_args()


def main():
    current_path = pathlib.Path(__file__).parent.resolve()
    args.output_dir = os.path.join(current_path, "output_segmentation")
    args.cont = os.path.join(current_path, "container_segmentation")

    opt_blur_kernel_size = 11
    opt_sigma = 20
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(",")]
    assert len(gpus) == 1
    if not args.gpu == "None":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = 18
    input_size = [512, 512]

    model = networks.init_model("resnet101", num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)["state_dict"]
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model.cuda()
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
        ]
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.container):
        os.makedirs(args.container)

    head_palette = [
        item for sublist in [[0] * 11 * 3, [255] * 3, [0] * 6 * 3] for item in sublist
    ]
    palette = [
        item
        for sublist in [[0] * 12, [255] * 3, [0] * 9 * 3, [255] * 6, [0] * 6]
        for item in sublist
    ]

    while True:
        if os.listdir(args.input_dir) == []:
            time.sleep(10)
        else:
            dataset = SimpleFolderDataset(
                root=args.input_dir, input_size=input_size, transform=transform
            )
            dataloader = DataLoader(dataset)
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dataloader)):
                    image, meta = batch
                    img_name = meta["name"][0]
                    c = meta["center"].numpy()[0]
                    s = meta["scale"].numpy()[0]
                    w = meta["width"].numpy()[0]
                    h = meta["height"].numpy()[0]

                    output = model(image.cuda())
                    upsample = torch.nn.Upsample(
                        size=input_size, mode="bilinear", align_corners=True
                    )
                    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                    upsample_output = upsample_output.squeeze()
                    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                    logits_result = transform_logits(
                        upsample_output.data.cpu().numpy(),
                        c,
                        s,
                        w,
                        h,
                        input_size=input_size,
                    )
                    parsing_result = np.argmax(logits_result, axis=2)
                    parsing_result_path = os.path.join(
                        args.output_dir, img_name[:-4] + ".png"
                    )
                    save_to_container_path = os.path.join(
                        args.cont, img_name[:-4] + ".png"
                    )
                    cont_image = preproc_img_size(
                        Image.open(os.path.join(args.input_dir, img_name))
                    )
                    cont_image.save(save_to_container_path)
                    os.remove(os.path.join(args.input_dir, img_name))
                    output_img = Image.fromarray(
                        np.asarray(parsing_result, dtype=np.uint8)
                    )
                    head_img = output_img.copy()
                    head_img.putpalette(head_palette)
                    output_img.putpalette(palette)
                    dil_param = img_name[:-4].split(sep="_")[-1]
                    output_img = dilation(output_img.convert("RGB"), int(dil_param))
                    output_img = remove_head_from_mask(
                        output_img.convert("RGB"), head_img.convert("RGB")
                    )
                    output_img = blur(
                        img=output_img,
                        blur_kernel_size=opt_blur_kernel_size,
                        sigma=opt_sigma,
                    )
                    output_img = preproc_img_size(output_img)
                    output_img.save(parsing_result_path)
                    if img_name[:-4] == "stop_the_process":
                        return


if __name__ == "__main__":
    main()
