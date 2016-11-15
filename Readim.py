import skimage
import skimage.io
import skimage.transform

import os
from glob import glob

import numpy as np
import scipy.io

class MyImages:
    
    def __ini__(self):
        pass
        
    def build(self,im_dir,batch_size,ext='png'):
        ext = '*.'+ext
        self.train_urls = os.path.join(im_dir,'train',ext)
        self.label_urls = os.path.join(im_dir,'label',ext)
        
        self.train_imgs = self.readims(self.train_urls)
        self.label_imgs = self.readims(self.label_urls)
        self.label_imgs = self.label_parsing(self.label_imgs)

        self.start = 0
        self.end = batch_size
        self.batch_size = batch_size
        
    def readims(self,path):
        c = skimage.io.ImageCollection(glob(path))
        all_images = c.concatenate()
        
        if all_images.ndim == 3:
            all_images=np.reshape(all_images, all_images.shape + (1,))
            #all_images=all_images[...,np.newaxis]
            
        all_images = all_images.astype('float32')
        
        #scipy.io.savemat('imgs.mat', mdict={'imgs': all_images})
        return all_images
        
    def label_parsing(self,images):
        #images = np.concatenate((images,images),axis=3)
        images[images>0] = 1
        #images[np.where( images >= 100 )] = 1
        #images = self.not_label(images)        
        
        return images
        
    def not_label(self,images):
        tmp = images[:,:,:,1]
        tmp[ tmp == 0 ] =-1
        tmp[ tmp == 1 ] = 0
        tmp[ tmp ==-1 ] = 1
        images[:,:,:,1] = tmp

        return images
        
    def nextBatch(self):
        size = self.train_imgs.shape[0]
        batch_images = None
        
        if self.end > self.start:
            batch_images = self.train_imgs[self.start:self.end,:,:,:]
            batch_labels = self.label_imgs[self.start:self.end,:,:,:]

        else:
            tmp1 = self.train_imgs[self.start:size,:,:,:]
            tmp2 = self.train_imgs[0:self.end,:,:,:]
            batch_images = np.concatenate((tmp1,tmp2),axis=0)
            
            tmp3 = self.label_imgs[self.start:size,:,:,:]
            tmp4 = self.label_imgs[0:self.end,:,:,:]
            batch_labels = np.concatenate((tmp3,tmp4),axis=0)
            
        while True:
            self.start = (self.start + self.batch_size) % size
            self.end = (self.end + self.batch_size) % size
            if not self.start>= size+1 and not self.end == 0:
                break
        
        #scipy.io.savemat('batch_images.mat', mdict={'bimgs': batch_images})
        return batch_images, batch_labels
