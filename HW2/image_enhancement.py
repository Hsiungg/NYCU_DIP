import cv2
import numpy as np


"""
TODO Part 1: Gamma correction
"""
def gamma_correction(input_img, gamma):
    output_img = np.power(input_img / 255.0, gamma) * 255.0
    return output_img.astype(np.uint8)

"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(input_img):
    hsv_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    hist, bins = np.histogram(v_channel.flatten(), bins=256, range=[0,256], density=True)
    cdf = hist.cumsum()
    cdf -= cdf.min()
    cdf = (255 * cdf).astype('uint8')
    equalized_v= cdf[v_channel]
    hsv_image[:, :, 2] = equalized_v
    output_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return output_img

"""
Bonus
"""
#CLAHE (Contrast Limited Adaptive Histogram Equalization)
def other_enhancement_algorithm(input_img, cliplimit = 7.0, gridsize = (8, 8)):
    hsv_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]
    h, w = v_channel.shape
    h_tile_size, w_tile_size = h // gridsize[0], w // gridsize[1]
    output_tiles = np.zeros((gridsize[0], gridsize[1], h_tile_size, w_tile_size), dtype=np.uint8)
    output_v_channel = np.zeros_like(v_channel, dtype=np.uint8)
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            h_start, h_end = i * h_tile_size, (i + 1) * h_tile_size
            w_start, w_end = j * w_tile_size, (j + 1) * w_tile_size
            tile = v_channel[h_start:h_end, w_start:w_end]
            hist, bins = np.histogram(tile.flatten(), bins=256, range=[0,256])
            clip_value = int(cliplimit * tile.size / 256)
            clipped_hist = np.clip(hist, 0, clip_value)
            excess_sum = hist.sum() - clipped_hist.sum()
            add_back = excess_sum // 256
            clipped_hist += add_back
            cdf = clipped_hist.cumsum()
            cdf = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min())
            cdf = cdf.astype('uint8')
            output_v_channel[i * h_tile_size: (i + 1) * h_tile_size, j * w_tile_size: (j + 1) * w_tile_size] = cdf[tile].reshape(h_tile_size, w_tile_size)
    hsv_image[:, :, 2] = output_v_channel
    output_img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return output_img


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")
    # shape = (384, 512, 3)
    # TODO: modify the hyperparameter
    gamma_list = [0.2, 0.45, 1.5] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, gamma)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)

    other_img= other_enhancement_algorithm(img)
    cv2.imshow("CLAHE", np.vstack([img, other_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
