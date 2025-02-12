import cv2
import numpy as np
import os
from mark_max import mark_max_points
"""
TODO White patch algorithm
"""
def white_patch_algorithm(img):
    img_float = img.astype(np.float32)
    BGR_max = np.max(img_float, axis=(0, 1))
    corrected_image = img_float / BGR_max * 255.0
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image

"""
TODO Gray-world algorithm
"""
def gray_world_algorithm(img):
    img_float = img.astype(np.float32)
    BGR_avg = np.mean(img_float, axis=(0, 1))
    gray_avg = np.mean(BGR_avg)
    BGR_scale = gray_avg / BGR_avg
    corrected_image = img_float * BGR_scale
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image

"""
Bonus 
"""
def shades_of_gray_algorithm(img, p = 6):
    img = img.astype(np.float32)
    BGR_norm = np.linalg.norm(img.reshape(-1, 3), ord=p, axis=0)
    BGR_scale = BGR_norm / np.mean(BGR_norm)
    corrected_image = np.zeros_like(img)
    for i in range(3):
        corrected_image[:,:, i] = img[:,:, i] / BGR_scale[i]
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image
"""
Main function
"""
def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))
        marked_image = mark_max_points(img)
        # TODO White-balance algorithm
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)
        shades_of_gray_img = shades_of_gray_algorithm(img)

        cv2.imwrite("result/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("result/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)
        cv2.imwrite("result/color_correction/shade_of_gray_input{}.bmp".format(i + 1), shades_of_gray_img)
        cv2.imwrite("result/color_correction/input_marked{}.bmp".format(i + 1), marked_image)

if __name__ == "__main__":
    main()