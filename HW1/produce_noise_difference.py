import numpy as np
import cv2

if __name__ == "__main__":
    input_img_1 = cv2.imread("output_median_5.jpg")
    input_img_2 = cv2.imread("input_part1.jpg")
    diff = input_img_1 - input_img_2
    cv2.imwrite("diff_1.jpg", diff)