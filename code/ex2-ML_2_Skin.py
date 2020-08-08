import sys, os
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from myMVND import MVND
from imageHelper import imageHelper
from classifyHelper import classify, get_prior

#matplotlib.use('TkAgg')

dataPath = '../data/'


def mvndSkinDetection() -> None:
    '''
    Skin detection - compute a MVND for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']

    mvn_sskin = [MVND(sdata)]
    mvn_nskin = [MVND(ndata)]
    # Optain priors
    mask = imageHelper()
    mask.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    # TODO: EXERCISE 2 - Compute the skin and nonskin prior
    prior_skin, prior_nonskin = get_prior(mask)

    print("TRAINING DATA")
    trainingmaskObj = imageHelper()
    trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    trainingimageObj = imageHelper()
    trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    classify(trainingimageObj, trainingmaskObj, mvn_sskin, mvn_nskin, "Training", prior_skin, prior_nonskin)

    print("TEST DATA")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test.png'))
    classify(testimageObj, testmaskObj, mvn_sskin, mvn_nskin, "Test", prior_skin, prior_nonskin)
    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nMVND exercise - Skin detection")
    print("##########-##########-##########")
    mvndSkinDetection()
    print("##########-##########-##########")