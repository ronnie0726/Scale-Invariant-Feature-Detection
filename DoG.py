import numpy as np
import cv2
from sqlalchemy import false, true


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.printdog = None
        self.picture = '1'

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images.append(image)
        height, width= image.shape
        for o in range(self.num_octaves):
            if o != 0:
                img = cv2.resize(gaussian_images[self.num_DoG_images_per_octave*o], (width//2, height//2), interpolation=cv2.INTER_NEAREST)
                gaussian_images.append(img)
            else:
                img = image
            for i in range(self.num_DoG_images_per_octave):
                gaussian_images.append(cv2.GaussianBlur (img, (0, 0), self.sigma**(1 + i)))


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(0, self.num_DoG_images_per_octave):
            dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))
        for i in range(self.num_DoG_images_per_octave+1, self.num_DoG_images_per_octave+5):
            dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))         

        if self.printdog:
            for i in range(len(dog_images)):
                img = np.zeros(shape = (dog_images[i].shape[0],dog_images[i].shape[1]))
                cv2.normalize(dog_images[i],img,0,255,cv2.NORM_MINMAX)
                cv2.imwrite('./Dog Image/'+self.picture+'-'+str(i+1)+".png", img)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        def iskey(top,mid,buttom,i,j):
            if (abs(mid[i][j]) <= self.threshold):
                return False
            l = []
            for x in range(-1,2):
                for y in range(-1,2):
                    l.append(top[i+x][j+y])
                    l.append(mid[i+x][j+y])
                    l.append(buttom[i+x][j+y])
            if mid[i][j] >= max(l) or mid[i][j] <= min(l):
                return True
            else:
                return False
        


        keypoints = np.array([])
        for x in range(0,2):
            top = dog_images[x+2]
            mid = dog_images[x+1]
            buttom = dog_images[x]
            for i in range(1,height-1):
                for j in range(1,width-1):
                    if (iskey(top,mid,buttom,i,j)):
                        keypoints = np.append(keypoints, np.array([i,j]))

        for x in range(4,6):
            top = dog_images[x+2]
            mid = dog_images[x+1]
            buttom = dog_images[x]
            for i in range(1,height//2-1):
                for j in range(1,width//2-1):
                    if (iskey(top,mid,buttom,i,j)):
                        keypoints =np.append(keypoints, np.array([2*i,2*j]))
        keypoints = keypoints.reshape(-1,2)


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
