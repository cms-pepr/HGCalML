

import skimage
from skimage import color
import matplotlib.pyplot as plt
import numpy as np

import random
#generate one shape at a time and then add the numpy arrays only where there are zeros

npixel=128


def getCenterCoords(t1):
    coords1a = (t1[0][1][0][1]+t1[0][1][0][0])/2.
    coords1b = (t1[0][1][1][1]+t1[0][1][1][0])/2.
    return coords1a, coords1b

def labeltoonehot(desc):
    if desc[0][0] == 'rectangle':
        return np.array([1,0,0],dtype='float32')
    if desc[0][0] == 'triangle':
        return np.array([0,1,0],dtype='float32')
    if desc[0][0] == 'circle':
        return np.array([0,0,1],dtype='float32')
    
    raise Exception("labeltoonehot: "+desc[0][0])
   
def createPixelTruth(desc,image, ptruth):
    onehot = labeltoonehot(desc)
    coordsa,coordsb = getCenterCoords(desc)
    coords = np.array([coordsa,coordsb], dtype='float32')
    truth =  np.concatenate([coords,onehot])
    truth = np.expand_dims(np.expand_dims(truth, axis=0),axis=0)
    truth = np.tile(truth, [image.shape[0],image.shape[1],1])
    if ptruth is None:
        ptruth = np.zeros_like(truth)+255
    truthinfo = np.where(np.tile(np.expand_dims(image[:,:,0]<255, axis=2), [1,1,truth.shape[2]]), truth, ptruth)
    return truthinfo
    

   
    
    
def checktuple_overlap(t1,t2):
    x1,y1 = getCenterCoords(t1)
    x2,y2 = getCenterCoords(t2)
    diff = (x1-x2)**2 + (y1-y2)**2
    if diff < 30:
        return True
    return False
    
    
def checkobj_overlap(dscs,dsc):
    for d in dscs:
        if checktuple_overlap(dsc,d):
            return True
    return False
    

def generate_shape():
    image, desc = skimage.draw.random_shapes((npixel, npixel),  max_shapes=1, 
                                      min_size=npixel/5, max_size=npixel/3,
                                      intensity_range=((100, 254),))
    return image, desc

def addshape(image , desclist):
    if image is None:
        image, d = generate_shape()
        return image, image, d, True
    
    new_image, desc = generate_shape()
    
    if checkobj_overlap(desclist,desc):
        return image, image, desc, False
    
    shape = new_image.shape
    select = new_image > 254
    
    return np.where(select,image,new_image), new_image, desc, True


def create_images(nimages = 1000):
    alltruth = []
    allimages = []
    for e in range(nimages):
        dsc=[]
        image=None
        nobjects = random.randint(1,7)
        addswitch=True
        indivimgs=[]
        indivdesc=[]
        ptruth=None
        for i in range(nobjects):
            new_image, obj, des, addswitch = addshape(image,indivdesc)
            if addswitch:
                ptruth = createPixelTruth(des, obj, ptruth)
                image = new_image
        
        
        image = np.expand_dims(image,axis=0)
        ptruth = np.expand_dims(ptruth,axis=0)
        allimages.append(image)
        alltruth.append(ptruth)
    
    return np.concatenate(allimages, axis=0), np.concatenate(alltruth,axis=0)
