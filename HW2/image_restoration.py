import cv2
import numpy as np
import math

"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(length, angle, img):
    psf = np.zeros((length, length), dtype=np.float32)
    center = (psf.shape[0] // 2, psf.shape[1] // 2)
    end = (int(center[0] - length * np.cos(np.deg2rad(angle) ) ),
           int(center[1] + length * np.sin(np.deg2rad(angle) ) ))
    cv2.line(psf, center, end, (255,), thickness=1)
    psf /= psf.sum()
    return psf
"""
TODO Part 2: Wiener filtering
"""
def wiener_filtering(img, psf, K):
    out_img = np.zeros_like(img)
    for i in range(3):
        channel = img[:,:,i]
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, channel.shape)
        wiener_filter = np.conj(H) / (np.abs(H)**2 + K)
        f_channel = wiener_filter * G 
        out_channel = np.fft.ifft2(f_channel)
        out_channel = np.abs(out_channel).astype(np.float32)
        out_channel = np.clip(out_channel, 0, 255)
        out_channel = out_channel.astype(np.uint8)
        out_img[:,:,i] = out_channel
    return out_img

"""
TODO Part 3: Constrained least squares filtering
"""
def constrained_least_square_filtering(img, psf, gamma):
    out_img = np.zeros_like(img)
    P = np.fft.fft2([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]], 
                    out_img[:,:,0].shape)
    for i in range(3):
        channel = img[:,:,i]
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, channel.shape)
        cls_filter = np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2 )
        f_channel = cls_filter * G
        out_channel = np.fft.ifft2(f_channel)
        out_channel = np.abs(out_channel).astype(np.float32)
        out_channel = np.clip(out_channel, 0, 255)
        out_channel = out_channel.astype(np.uint8)
        out_img[:,:,i] = out_channel
    return out_img

"""
Bouns
"""
#Richardson-Lucy
def other_restoration_algorithm(img, iter, length, angle):
    out_img = img
    psf = generate_motion_blur_psf(length, angle, img)
    neg_psf = generate_motion_blur_psf(length, -angle, img)
    for i in range(iter):
        out_img = np.zeros_like(img)
        for j in range(3):
            relative_blur = cv2.filter2D(out_img[:,:,j], -1, psf)
            relative_blur = np.where(relative_blur == 0, 1e-8, relative_blur)
            out_img[:,:,j] *= cv2.filter2D(out_img[:,:,j] / relative_blur, -1, neg_psf).astype(np.uint8)
            out_img[:,:,j] = np.clip(out_img[:,:,j], 0, 255).astype(np.uint8)
    return out_img

def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr

"""
Main function
"""
def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))
        # TODO Part 1: Motion blur PSF generation
        length = 55
        angle = 45
        K = 0.007
        psf = generate_motion_blur_psf(length, angle, img_blurred)
        # TODO Part 2: Wiener filtering
        wiener_img = wiener_filtering(img_blurred, psf, K)
        y_length = int(length * np.cos(np.deg2rad(angle)))
        x_length = length - int(length * np.sin(np.deg2rad(angle))) - 4
        wiener_img = np.vstack((wiener_img[-y_length:], wiener_img[:-y_length]))
        wiener_img = np.hstack((wiener_img[:, -x_length:], wiener_img[:, :-x_length]))
        # TODO Part 3: Constrained least squares filtering
        constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, 3.5)
        constrained_least_square_img = np.vstack((constrained_least_square_img[-y_length:], constrained_least_square_img[:-y_length]))
        constrained_least_square_img = np.hstack((constrained_least_square_img[:, -x_length:], constrained_least_square_img[:, :-x_length]))
        richardson_lucy_img = other_restoration_algorithm(img_blurred, 1, 3, 45)
        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, constrained_least_square_img)))

        cv2.imshow("window", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
