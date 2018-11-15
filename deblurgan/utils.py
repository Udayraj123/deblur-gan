import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf


RESHAPE = (256,256)



def high_freq(img):

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    rows, cols = img.shape
    crow,ccol = int(rows/2) , int(cols/2)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_high = cv2.idft(f_ishift)
    img_high = cv2.magnitude(img_high[:,:,0],img_high[:,:,1])

    mask2 = np.zeros((rows,cols,2),np.uint8)
    mask2[crow-30:crow+30, ccol-30:ccol+30] = 1
    
    # apply mask and inverse DFT
    fshift = dft_shift*mask2
    f_ishift = np.fft.ifftshift(fshift)
    img_low = cv2.idft(f_ishift)
    img_low = cv2.magnitude(img_low[:,:,0],img_low[:,:,1])

    # from matplotlib import pyplot as plt
    # plt.subplot(121),plt.imshow(img_high, cmap = 'gray')
    # plt.title('Frequency Component'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_low, cmap = 'gray')
    # plt.title('Base Component'), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    # img_high = cv2.cvtColor(img_high, cv2.COLOR_GRAY2BGR)
    return img_high,img_low
    
def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
