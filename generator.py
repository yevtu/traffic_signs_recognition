import numpy as np
import cv2
from scipy.ndimage.interpolation import shift, rotate


def get_augmented_image(path, target_size=(32, 32)):
    img = cv2.imread(path)
    img = rotate(img, np.random.randint(-15, 15), reshape=False)
    img = shift(img, [np.random.randint(-5, 5), np.random.randint(-5, 5), 0])


    def random_brightness(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] *= (.25 + np.random.uniform())
        return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


    img = random_brightness(img)
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def get_label(path, n_classes=42):
    cls = int(path.split('/')[-2])
    one_hot = np.zeros((n_classes))
    one_hot[cls] = 1
    return one_hot


def generator(imgpaths, n_classes=42, batch_size=32):
    k = 0
    while True:
        X = np.zeros((batch_size, 32, 32, 3))
        y = np.zeros((batch_size, n_classes))

        for i in range(batch_size):
            k += 1
            if not k % len(imgpaths):
                np.random.shuffle(imgpaths)

            X[i] = get_augmented_image(imgpaths[k], target_size=(32, 32))
            y[i] = get_label(imgpaths[k])

        yield X, y