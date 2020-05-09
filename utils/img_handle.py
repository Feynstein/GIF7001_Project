import cv2
import numpy as np
import urllib
import os
from tkinter import messagebox
import tkinter as tk


class GetImg:

    def __init__(self, path):
        self.img = None
        self.crop = None
        self.path = path
        self.load__()

        self.refPt = None
        self.cropping = None

    def __call__(self):
        return self.img

    def resize(self, ratio=None, wide=None):
        img = self.img
        h, w = img.shape[:2]
        if ratio is not None and wide is not None:
            print('Resize has failed, choose whether ratio or wide not both')
        elif wide is not None:
            ratio = wide / w
        else:
            pass
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_CUBIC)
        self.img = img

    def load__(self):
        path = self.path
        if path is None:
            raise Exception("First specify imPath with imData.imPath = \'...\'")

        if 'http' in path:
            with urllib.request.urlopen(path) as url:
                resp = url.read()
            img = np.asarray(bytearray(resp), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path)
        if img is None:
            raise Exception('Image failed to be loaded, current working dir is {}'.format(os.getcwd()))
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def click_and_crop__(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        name = "Template: Selectionnez le crop a effectuer sur cette image"

        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.cropping = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            self.cropping = False

            # draw a rectangle around the region of interest
            temp_img = self.crop.copy()
            cv2.rectangle(temp_img, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow(name, temp_img)

        elif len(self.refPt) == 1:
            cv2.imshow(name, self.crop)
            temp_img = self.crop.copy()
            cv2.rectangle(temp_img, self.refPt[0], (x,y), (0, 255, 0), 2)
            cv2.imshow(name, temp_img)

    def cropNsave(self, save=False):
        # initialize the list of reference points and boolean indicating
        # whether cropping is being performed or not
        self.refPt = []
        self.cropping = False

        self.crop = self.img.copy()
        name = "Template: Selectionnez le crop a effectuer sur cette image"
        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.click_and_crop__)

        # keep looping until the points are set
        cv2.imshow(name, self.crop)
        while True:
            key = cv2.waitKey(1) & 0xFF

            # if the 'c' or 'ESC' key is pressed, break from the loop
            if key == ord("c") or key == 27 or cv2.getWindowProperty(name, 0) < 0:
                break

            # if the rectangle is defined, break from the loop
            elif len(self.refPt) ==2:
                break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(self.refPt) == 2:
            self.crop = self.crop[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]

            if save:
                cv2.imshow("ROI", self.crop)
                root = tk.Tk()
                root.withdraw()
                result = messagebox.askokcancel("Python", "Would you like to save the data?")
                if result:
                    img_path = self.path
                    croped_filename = img_path[:-4] + '_crop' + img_path[-4:]
                    cv2.imwrite(croped_filename, self.crop);
                    print('Saved as {}'.format(croped_filename))
                else:
                    print('Not saved')

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    # if the 'c' or 'ESC' key is pressed, break from the loop
                    if key == ord("c") or key == 27 or cv2.getWindowProperty("ROI", 0) < 0:
                        break
        # close window
        cv2.destroyWindow(name)
        #cv2.destroyAllWindows()

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


if __name__ == '__main__':
    print(os.getcwd())
    img = GetImg('..//images//IMG_0116.JPG')
    img.resize(ratio=1/4)
    img.cropNsave()
    #print(img())
