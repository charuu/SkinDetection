import numpy as np
import imageio


class imageHelper:
    def __init__(self):
        pass

    def loadImage(self, image):
        self.image = image
        self.shape = self.image.shape
        self.dim = len(self.shape)
        self.N = self.shape[0]
        self.M = self.shape[1]
        try:
            self.D = self.shape[2]
        except:
            # Assume that we have 1 dimension
            self.D = 1
        self.numask = self.N * self.M

    def loadImage1dBinary(self, image, N, M):
        self.loadImage(np.reshape(image, (N, M)))

    def loadImageFromFile(self, path):
        self.imagePath = path
        self.loadImage(np.array(imageio.imread(self.imagePath)) / 255.0)

    def getLinearImageBinary(self):
        return np.reshape(self.image, (self.numask, 1))

    def getLinearImage(self):
        return np.reshape(self.image, (self.numask, self.D))

    def getImageBinary(self):
        return np.reshape(self.image, (self.N, self.M))