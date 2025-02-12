import numpy as np
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size, mode = 'zero'):
    ############### YOUR CODE STARTS HERE ###############
    pad_size = kernel_size // 2
    padded_img = np.zeros((input_img.shape[0] +  2 * pad_size, input_img.shape[1] + 2 * pad_size, 3), dtype = input_img.dtype)
    padded_img[pad_size : padded_img.shape[0] - pad_size, pad_size : padded_img.shape[1] - pad_size, :] = input_img
    if mode == 'edge':
        padded_img[pad_size:pad_size + input_img.shape[0], :pad_size] = input_img[:, 0, np.newaxis]
        padded_img[pad_size:pad_size + input_img.shape[0], -pad_size:] = input_img[:, -1, np.newaxis]
        padded_img[:pad_size, pad_size:pad_size + input_img.shape[1]] = padded_img[pad_size:pad_size + 1, pad_size:pad_size + input_img.shape[1]]
        padded_img[-pad_size:, pad_size:pad_size + input_img.shape[1]] = padded_img[pad_size + input_img.shape[0] - 1:pad_size + input_img.shape[0], 
                                                                                    pad_size:pad_size + input_img.shape[1]]
    ############### YOUR CODE ENDS HERE #################
    return padded_img

def convolution(input_img, kernel, mode= 'zero'):
    ############### YOUR CODE STARTS HERE ###############
    kernel_size = kernel.shape[0]
    padded_img = padding(input_img, kernel_size, mode = mode)
    output_img = np.zeros_like(input_img)
    for channel in range(input_img.shape[2]):
        for height in range(input_img.shape[0]):
            for width in range(input_img.shape[1]):
                output_img[height, width, channel] = np.sum(padded_img[height : height + kernel_size, width : width + kernel_size, channel] * kernel)
    ############### YOUR CODE ENDS HERE #################
    return output_img

def gaussian_filter(input_img, kernel_size, std):
    ############### YOUR CODE STARTS HERE ###############
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * std**2))
    kernel /= (2 * np.pi * std ** 2)
    kernel /= kernel.sum()
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def median_filter(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############
    output_img = np.zeros_like(input_img)
    padded_img = padding(input_img, kernel_size, mode='edge')
    for channel in range(input_img.shape[2]):
        for height in range(input_img.shape[0]):
            for width in range(input_img.shape[1]):
                region = padded_img[height : height + kernel_size, width : width + kernel_size, channel]
                output_img[height, width, channel] = np.median(region)
    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img, filter_num):
    ############### YOUR CODE STARTS HERE ###############
    kernel_1 = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]) 
    kernel_2 = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    if filter_num == 1:
        kernel = kernel_1
    elif filter_num == 2:
        kernel = kernel_2
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel, mode = 'edge')

if __name__ == "__main__":
    args = parse_args()
    #################set gaussian kernel_size and std here ############
    gaussian_kernel_size = 7
    gaussian_std = 3
    #################set median kernel_size here#######################
    median_kernel_size = 5
    ####################set filter_num here ###########################
    laplacian_filter_num = 1
    ###################################################################
    if args.gaussian: 
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img, gaussian_kernel_size , gaussian_std)
        cv2.imwrite(f"output_gaussian_{gaussian_kernel_size}_{gaussian_std}.jpg", output_img)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img, median_kernel_size)
        cv2.imwrite(f"output_median_{median_kernel_size}.jpg", output_img)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img, laplacian_filter_num)
        cv2.imwrite(f"output_laplacian_{laplacian_filter_num}.jpg", output_img)
