from PIL import Image, ImageEnhance
import numpy as np 
import random
import math
import cv2
import pdb
import os


def random_rotate(image, degree = [-3, 3]):
    angle = np.random.randint(degree[0], degree[1])
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def random_shift(image):
    offset_x = random.randint(-10, 10)
    offset_y = random.randint(-10, 10)
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    image = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return image

def random_brightness(input_img, brightness_range = [-100, 100]):
    brightness = np.random.ranf()*(brightness_range[1] - brightness_range[0]) + brightness_range[0]
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    return buf


def random_fade_brightness(input_img, brightness_range = [0, 0]):
    _type = ["up-down", "down-up", "left-right", "right-left"]
    brightness = np.random.ranf()*(brightness_range[1] - brightness_range[0]) + brightness_range[0]
    brightness_type = np.random.randint(len(_type))
    w, h, d = np.shape(input_img)
    if brightness == 0:
        buf = input_img.copy()
    elif _type[brightness_type] in ["up-down", "down-up"]:
        if _type[brightness_type] == "up-down":
            fade = np.arange(w) * (brightness/w)
        else:
            fade = np.flip(np.arange(w) * (brightness/w), axis = 0)
        buf = input_img.copy()
        for k in range(d):
            for j in range(h):
                for i in range(w):
                    buf[i][j][k] = min(buf[i][j][k] + fade[i], 255)
    elif _type[brightness_type] in ["left-right", "right-left"]:
        if _type[brightness_type] == "left-right":
            fade = np.arange(h) * (brightness/h)
        else:
            fade = np.flip(np.arange(h) * (brightness/h), axis = 0)
        buf = input_img.copy()
        for k in range(d):
            for i in range(w):
                for j in range(h):
                    buf[i][j][k] = min(buf[i][j][k] + fade[j], 255)
    return buf


def random_point_brightness(input_img, brightness_range = [0, 0]):
    brightness = np.random.ranf()*(brightness_range[1] - brightness_range[0]) + brightness_range[0]
    w, h, d = np.shape(input_img)
    x = np.random.randint(w)
    y = np.random.randint(h)
    fade = np.flip(np.arange(max(x, y, w - x, h - y)), axis = 0)
    _filter = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            distance = int(math.sqrt((x - i)**2 + (y - j)**2))
            if distance < len(fade):
                _filter[i][j] = fade[distance]
    buf = input_img.copy()
    for k in range(d):
            for i in range(w):
                for j in range(h):
                    buf[i][j][k] = min(buf[i][j][k] + _filter[i][j], 255)
    return buf


def random_contrast(input_img, contrast_range = [0, 1]):
    contrast = np.random.ranf()*(contrast_range[1] - contrast_range[0]) + contrast_range[0]
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(input_img, alpha_c, input_img, 0, gamma_c)
    else:
        buf = input_img.copy()
    return buf


def random_saturate(image, saturation_range = [0.3, 1]):
    image = image.astype(np.uint8)
    saturation = np.random.ranf()*(saturation_range[1] - saturation_range[0]) + saturation_range[0]
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(image)
    enhanced_im = enhancer.enhance(saturation)
    return np.array(enhanced_im)


def random_channel_shift(image):
    channels = cv2.split(image)
    np.random.shuffle(channels)
    image = cv2.merge(channels)
    return image


def random_inhanced_color_by_channel(image, range_inhanced = [10, 60]):
    inhanced_val = np.random.ranf()*(range_inhanced[1] - range_inhanced[0]) + range_inhanced[0]
    channel = np.random.randint(3)
    image[:,:,channel] += int(inhanced_val)
    return image


def random_noise(image):
    noise_typ = random.choice(["gauss", "s&p", "poisson", "speckle"])
    if noise_typ == "gauss":
        row, col, ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        image = image + gauss
        return image
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        image = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        image[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        image[coords] = 0
        return image
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = np.random.poisson(image * vals) / float(vals)
        return image
    elif noise_typ =="speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        image = image + image * gauss
        return image


def random_incline(image):
    _type = random.choice(["left", "right"])
    rate = random.uniform(0, 0.5)
    if _type == "left":
        h, w, d = image.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        shift = int(math.tan(rate*(math.pi/2))*h)
        image = cv2.copyMakeBorder(image,0,0,0,shift,cv2.BORDER_CONSTANT,value=[0,0,0])
        pts2 = np.float32([[0, 0], [w, 0], [shift, h]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (w+shift, h))
    elif _type == "right":
        h, w, d = image.shape
        pts1 = np.float32([[0, h], [w, h], [0, 0]])
        shift = int(math.tan(rate*(math.pi/2))*h)
        image = cv2.copyMakeBorder(image,0,0,0,shift,cv2.BORDER_CONSTANT,value=[0,0,0])
        pts2 = np.float32([[0, h], [w, h], [shift, 0]])
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (w+shift, h))
    return image


def random_motion_blur(image):
    size = random.randint(5, 15)
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    # applying the kernel to the input image
    image = cv2.filter2D(image, -1, kernel_motion_blur)
    return image


filters = {
    'random_brightness': random_brightness, 
    #'random_shift': random_shift,
    #'random_channel_shift': random_channel_shift, 
    'random_contrast': random_contrast, 
    'random_fade_brightness': random_fade_brightness, 
    #'random_inhanced_color_by_channel': random_inhanced_color_by_channel, 
    'random_noise': random_noise, 
    # 'random_point_brightness': random_point_brightness, 
    # 'random_rotate': random_rotate, 
    'random_saturate': random_saturate,
    'random_incline': random_incline,
    'random_motion_blur': random_motion_blur
}

# image = cv2.imread("/home/dunglt/cmnd/po/data/valid/45_nq_8.png")
# cv2.imwrite("data/test/transform/transformed/rotation.jpg", random_rotate(image))
# cv2.imwrite("data/test/transform/transformed/shift.jpg", random_shift(image))
# cv2.imwrite("data/test/transform/transformed/brightness.jpg", random_brightness(image))
# cv2.imwrite("data/test/transform/transformed/fade_brightness.jpg", random_fade_brightness(image))
# cv2.imwrite("data/test/transform/transformed/point_brightness.jpg", random_point_brightness(image))
# cv2.imwrite("data/test/transform/transformed/contrast.jpg", random_contrast(image))
# cv2.imwrite("data/test/transform/transformed/saturate.jpg", random_saturate(random_noise(image)))
# cv2.imwrite("data/test/transform/transformed/channel_shift.jpg", random_channel_shift(image))
# cv2.imwrite("data/test/transform/transformed/inhanced_color_by_channel.jpg", random_inhanced_color_by_channel(image))
# cv2.imwrite("data/test/transform/transformed/noise.jpg", random_noise(image))
# cv2.imwrite("data/test/transform/transformed/incline.jpg", random_incline(image))
# cv2.imwrite("data/test/transform/transformed/motion_blur.jpg", random_motion_blur(image))


# cv2.imshow("image", random_saturate(image, [0.3, 0.3]))
# cv2.waitKey(0)
