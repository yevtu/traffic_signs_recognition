import numpy as np
import cv2
from scipy.ndimage.interpolation import shift, rotate


def random_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 2] *= (.25 + np.random.uniform())
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def get_image(path, augment, target_size=(32, 32)):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype('float32')
    img *= 1./255

    if augment:
        img = rotate(img, np.random.randint(-10, 10), reshape=False)
        img = shift(img, [np.random.randint(-3, 3), np.random.randint(-3, 3), 0])
        img = random_brightness(img)

    img = 2 * img - 1
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def get_label(path, n_classes=42):
    cls = int(path.split('/')[-2])
    one_hot = np.zeros((n_classes))
    one_hot[cls] = 1
    return one_hot


def generator(imgpaths, n_classes=42, batch_size=32, augment=False):
    k = 0
    while True:
        X = np.zeros((batch_size, 32, 32, 3), dtype=np.float)
        y = np.zeros((batch_size, n_classes))

        for i in range(batch_size):
            k = (k + 1) % len(imgpaths)
            if not k:
                np.random.shuffle(imgpaths)

            X[i] = get_image(imgpaths[k], augment, target_size=(32, 32))
            y[i] = get_label(imgpaths[k], n_classes=n_classes)

        yield X, y