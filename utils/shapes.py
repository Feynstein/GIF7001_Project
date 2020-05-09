import cv2
import numpy as np
from utils.graphics import pltshow
from skimage import exposure
from skimage import feature


class FindShapes:

    def __init__(self, img):
        self.originalImg = img
        self.img = img.copy()
        self.mask = None
        self.img_masked = None
        self.edges = None
        self.smooth = None
        self.edgesAndContours = None
        self.contour_list = None
        self.houghedges = None
        self.hogImage = None

    def gaus_kernel(self, kernel_size, sigma=None):
        if sigma is None:
            sigma = kernel_size / 10

        # Output un filtre linéairement séparé (fonctionne avec sepfilter2D)
        kernel = cv2.getGaussianKernel(kernel_size, sigma, cv2.CV_32F)

        kernel = kernel.T * kernel
        return kernel

    def draw_lines(self, img, lines):
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)

    def extractMask(self, color):
        img = self.img
        colors = {'green': (np.array([40, 90, 40]), np.array([80, 255, 255]))}

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # OpenCV uses  H: 0 - 180, S: 0 - 255, V: 0 - 255
        if color in colors:
            lower_green, upper_green = colors[color]
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Filtre morphologique
            auto_size = int(np.floor(self.img.shape[0]/100 / 2.) * 2 + 1)
            auto_size = max(auto_size, 3)
            print('auto_size for mask: {}'.format(auto_size))
            kernel_morph = np.ones((auto_size, auto_size), np.uint8)
            mask = 255 - cv2.morphologyEx(255 - mask, cv2.MORPH_OPEN, kernel_morph)
            mask = cv2.medianBlur(mask, auto_size)

            # On retire la couleur verte:
            res = cv2.bitwise_and(img, img, mask=mask)

            self.mask = mask
            self.img_masked = img - res
        else:
            print("Mask hasn't been extracted because the {} color is not yet supported".format(color))

    def find_edges(self):
        img = self.img

        if self.img_masked is not None:
            gray = cv2.cvtColor(self.img_masked, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        auto_size = int(np.floor(self.img.shape[0] / 500 / 2.) * 2 + 1)
        auto_size = max(auto_size, 3)
        print('auto_size for edges: {}'.format(auto_size))
        kernel = self.gaus_kernel(auto_size, sigma=(auto_size+2)/4)
        gray = cv2.filter2D(gray, -1, kernel, borderType=cv2.BORDER_DEFAULT)
        # gray = cv2.medianBlur(gray,5)
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        self.edges = edges

    def smooth_edges(self, spread_value=1):
        # Ajoute des edges gras afin d'augmenter l'erreur admissible à la forme
        edges = self.edges
        auto_size = int(np.floor(self.edges.shape[0]*spread_value / 600 / 2.) * 2 + 1)
        print('auto_size for edges spreading: {}'.format(auto_size))
        kernel = self.gaus_kernel(auto_size +2, sigma=(auto_size +2)/16)
        gray = cv2.filter2D(edges, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Avec le filtre Guassien, la valeur max de l'image diminue si elle n'est pas toute blanche,
        # Il est préférable de rapporter cette valeur à 255
        dst = np.zeros((3, 3))
        gray = cv2.normalize(gray, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.smooth = gray

    def houghLines(self):
        self.houghedges = self.img.copy()
        minLineLength = 100
        maxLineGap = 1
        lines = cv2.HoughLinesP(self.edges, 1, np.pi / 180, 40, minLineLength, maxLineGap)
        thick = int(self.houghedges.shape[0] / 300)
        thick = max(thick, 1)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(self.houghedges, (x1, y1), (x2, y2), (0, 0, 0), thick)

    def HOG(self):
        if self.img_masked is not None:
            img = self.img_masked
        else:
            img = self.img.copy()
        hogImage = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

        # extract Histogram of Oriented Gradients
        (H, hogImage) = feature.hog(hogImage, orientations=9, pixels_per_cell=(7, 7),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        self.hogImage = hogImage

    def show(self):
        showDict1 = {1:self.img_masked, 2:self.mask, 3:self.edges}
        images_fig1 = tuple([image for key, image in showDict1.items() if image is not None])
        showDict2 = {4:self.edgesAndContours, 5:self.houghedges, 6:self.hogImage}
        images_fig2 = tuple([image for key, image in showDict2.items() if image is not None])
        pltshow(images_fig1, figsize=(8, 8))
        pltshow(images_fig2, figsize=(10, 10))