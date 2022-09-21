import sys
import timeit
import numpy as np
import cv2

def bright_pass(img, threshold):
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask[:, :, 1] = np.where(mask[:, :, 1] > threshold, mask[:, :, 1], 0)
    return cv2.cvtColor(mask, cv2.COLOR_HLS2BGR)

def approximated_gaussian_blur(img, sigma, n_iterations):
    w_ideal = np.sqrt((12*(sigma**2)/n_iterations) + 1)
    w_l = (int)((w_ideal//2)*2 + 1)
    w_u = w_l + 2
    m = (int)((12*(sigma**2) - n_iterations*(w_l**2) - 4*n_iterations*w_l - 3*n_iterations)/(-4*w_l - 4))
    for i in range(m):
        img = cv2.blur(img, (w_l, w_l))
    for i in range(n_iterations - m):
        img = cv2.blur(img, (w_u, w_u))
    return img

def gaussian_bloom(mask, iterations, initial_sigma):
    new_mask = np.zeros_like(mask)
    for i in range(iterations):
        new_mask += cv2.GaussianBlur(mask, (0, 0), initial_sigma*(2**i))
    return new_mask

def averages_bloom(mask, iterations, initial_sigma):
    new_mask = np.zeros_like(mask)
    for i in range(iterations): 
        new_mask += approximated_gaussian_blur(mask, initial_sigma*(2**i), 5)
    return new_mask

INPUT_IMAGES = (
    'GT2.BMP',
    'Wind Waker GC.bmp',
)

BLOOMS = (
    ("gaussian",   gaussian_bloom),
    ("averages",   averages_bloom),
)

def main ():
    for name in INPUT_IMAGES:
        img = cv2.imread (name, cv2.IMREAD_COLOR)
        if img is None:
            print ('Failed to open image. \n')
            sys.exit ()

        img = img.astype (np.float32) / 255

        for bloom_name, bloom in (BLOOMS):
            start_time = timeit.default_timer()

            mask = bright_pass(img, 0.5)
            mask = bloom(mask, 4, 1)

            out = 0.7*img + 0.3*mask

            print (f'Time {name} {bloom_name} : {timeit.default_timer () - start_time}')

            cv2.imwrite (f'{name}-bloom-{bloom_name}.png', out*255)

if __name__ == '__main__':
    main ()