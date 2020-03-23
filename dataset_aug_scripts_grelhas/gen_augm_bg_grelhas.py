'''
python gen_augm_bg_grelhas.py ../imgs/Grelhas/JPEGImages/ ../imgs/Grelhas/mask/ ../dtd/images/ ../test/
'''
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import glob 
import cv2
import sys
from tqdm import tqdm


image_list = []
mask_list = []
label_list = []
bkg_list = []

path_img = glob.glob(sys.argv[1]+'/*.jpg')
path_img.sort()
path_mask =glob.glob(sys.argv[2]+'/*.png')
path_mask.sort()
path_bg =glob.glob(sys.argv[3]+'/*/*.jpg')
path_bg.sort()
path_to_save_files = sys.argv[4]



sometimes = lambda aug: iaa.Sometimes(0.5, aug)

ia.seed(1)



# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # # crop images by -5% to 10% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.05, 0.1),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL, # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            fit_output=True # the output will fit the new object, so in those cases the image does not end up outside the image frame
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)
i =0
for idx_bg,_ in tqdm(enumerate(path_bg)):
    for idx_fg , _ in tqdm(enumerate(path_img)):
        fg = im=cv2.imread(path_img[idx_fg])
        bg = im=cv2.imread(path_bg[idx_bg])
        idx_bg+=1
        im_h,im_w,_ = fg.shape
        bg = cv2.resize(bg,(im_w,im_h),interpolation = cv2.INTER_AREA)

        ret, mask = cv2.threshold(cv2.cvtColor(cv2.imread(path_mask[idx_fg]), cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # get first masked value (foreground)
        fg1 = cv2.bitwise_or(fg, fg, mask=mask)

        # get second masked value (background) mask must be inverted
        bg1 = cv2.bitwise_or(bg, bg, mask=mask_inv)

        # combine foreground+background
        new_img = cv2.bitwise_or(fg1, bg1)

        #BB
        x, y, w, h = cv2.boundingRect(mask)
        bb_ia = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)], shape=fg.shape)

        for r in range(5):
            #augmentation    
            seq_det = seq.to_deterministic()
            images_aug, bbs_aug = seq_det(image=new_img, bounding_boxes=bb_ia)

            
            bbs = bbs_aug[0]
            is_break = False

            if bbs.y1 < 0 or bbs.y1 >= im_h or bbs.x1 < 0 or bbs.x1 >= im_w:
                is_break = True
                break

            if bbs.y2 < 0 or bbs.y2 >= im_h or bbs.x2 < 0 or bbs.x2 >= im_w:
                is_break = True
                break
            if not (is_break):
                is_break = False        
                # Saving
                # print('saving augmented_{:05d}.png'.format(i))
                img_path = os.path.join(path_to_save_files, 'augmented_{:08d}.png'.format(i))
                cv2.imwrite(img_path, images_aug)
                img_y, img_x, _ = images_aug.shape
                text_file = open(path_to_save_files+'augmented_{:08d}.txt'.format(i), "wt")
                n = text_file.write('0 ' +str( float(bbs.x1)/float(img_x)) + ' ' + str( float(bbs.y1)/float(img_y)) + ' ' + str( (float(bbs.x2)-float(bbs.x1))/float(img_x)) + ' ' + str( (float(bbs.y2)-float(bbs.y1))/float(img_y)))

                # n = text_file.write('0 ' +str( float(x)/float(im_w)) + ' ' + str( float(y)/float(im_h)) + ' ' + str( float(w)/float(im_w)) + ' ' + str( float(h)/float(im_h)))
                text_file.close()
                i+=1

                # while(1):
                    
                #     img_viz = images_aug
                #     # cv2.rectangle(img_viz,(x,y),(x+w,y+h),(0,255,0),2)
                #     cv2.rectangle(img_viz,(int(bbs.x1),int(bbs.y1)),(int((bbs.x2)),int(bbs.y2)),(0,255,0),2)
                #     cv2.imshow('img',img_viz)
                #     k = cv2.waitKey(33)
                #     if k==27 or k==32:    # Esc key to stop
                #         break


       
            