#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import csv
import gzip
import numpy
import itertools
import SimpleITK
import radiomics
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import skimage
import skimage.io
import skimage.morphology
import argparse

FEATURE_INDEX = [7, 8, 9, 54, 73, 100, 103, 104, 147, 150]
"""
QUALITY_SCORE = lambda x: scipy.stats.norm.ppf(
    scipy.stats.ttest_1samp(x, 0.0).pvalue) if abs(numpy.sum(x)) > 0.0 else -4.0
"""
QUALITY_SCORE = lambda x: numpy.sum(x)
DELIMITER = ";"

def lung_mask(img):
    """ get separate segmented lungs """
    dsi = SimpleITK.RelabelComponent(
        SimpleITK.ConnectedComponent(SimpleITK.GetImageFromArray(img)), sortByObjectSize = True)

    try:    # left lung
        a = SimpleITK.GetArrayFromImage(dsi == 1)
        a[a == 1] = 255
    except:
        a = numpy.full(img.shape, 0, dtype = numpy.uint8)
        a[:, 0:int(a.shape[1] / 2.0)] = 255

    try:    # right lung
        b = SimpleITK.GetArrayFromImage(dsi == 2)
        b[b == 1] = 255
    except:
        b = numpy.full(img.shape, 0, dtype = numpy.uint8)
        b[:, int(b.shape[1] / 2.0):] = 255

    return(a, b)

def segment(image, mask, unet):
    """ lung segmentation algorithm """
    IM_SHAPE = (256, 256)
    x = numpy.expand_dims(skimage.exposure.equalize_hist(skimage.transform.resize(image, IM_SHAPE)), -1)
    y = numpy.expand_dims(skimage.transform.resize(mask, IM_SHAPE), -1)
    x = numpy.array([x])        # numpy array of images
    y = numpy.array([y])        # numpy array of images
    x -= x.mean()
    x /= x.std()
    img_data = ImageDataGenerator(rescale = 1.)

    for xx, yy in img_data.flow(x, y, batch_size = 1):
        """ flow runs an infinite loop; the output is the same """
        pred = unet.predict(xx)[..., 0].reshape(x[0].shape[:2]) > 0.5
        pred =skimage.morphology.remove_small_holes(
            skimage.morphology.remove_small_objects(
                pred, 0.02 * numpy.prod(IM_SHAPE)), 0.02 * numpy.prod(IM_SHAPE))
        pred = skimage.transform.resize(pred, image.shape) * 255
        return(pred.astype(numpy.uint8))

def feature(image, mask):
    fts = {}

    try:
        i = SimpleITK.GetImageFromArray(image)      # load image
        m = SimpleITK.GetImageFromArray(mask)       # load mask

        fts.update((radiomics.firstorder.RadiomicsFirstOrder(i, m)).execute())
        fts.update((radiomics.shape2D.RadiomicsShape2D(i, m)).execute())
        fts.update((radiomics.glcm.RadiomicsGLCM(i, m)).execute())
        fts.update((radiomics.gldm.RadiomicsGLDM(i, m)).execute())
        fts.update((radiomics.glrlm.RadiomicsGLRLM(i, m)).execute())
        fts.update((radiomics.glszm.RadiomicsGLSZM(i, m)).execute())
        fts.update((radiomics.ngtdm.RadiomicsNGTDM(i, m)).execute())
    except Exception as e:
        print(e)
        return(";".join(["0.0"] * 96))

    return(";".join([str(v.item()) for k, v in fts.items()]))

def run(fx, model = "trained_model.hdf5", mask = "4-mask.png"):
    """ find similar images based on query image """
    img = skimage.img_as_float(skimage.io.imread(fx, as_gray = True))
    x, y = lung_mask(segment(img,
        skimage.img_as_float(skimage.io.imread(mask, as_gray = True)),
        load_model(model)))
    s = f"{feature(img, skimage.img_as_float(x))}{DELIMITER}{feature(img, skimage.img_as_float(y))}"
    return(numpy.array(s.split(DELIMITER)).astype(float))   # query image radiomic features

def main(argv):
    tbl = {(i[0]).strip(): numpy.array(i[1:]).astype(float) for i in csv.reader(
        gzip.open(argv.index, mode = "rt"), delimiter = DELIMITER, quotechar = '"')}
    j = tbl[argv.image] if argv.image in tbl.keys() else run(argv.image, argv.model, argv.mask)
    r = {i: QUALITY_SCORE([abs(a - b) for (a, b) in zip(tbl[i][FEATURE_INDEX], j[FEATURE_INDEX])]) for i in tbl.keys()}
    q = {k: v for k, v in sorted(r.items(), key = lambda u: u[1], reverse = False)}
    print(dict(itertools.islice(q.items(), argv.max)))
    return(True)

if __name__ == "__main__":
    argv = argparse.ArgumentParser(description = """
        Search images with similar attributes.
        """, formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    argv.add_argument("--image", "-i", type = str, dest = "image",
        help = "[Required] Query image.")
    argv.add_argument("--index", "-x", type = str, dest = "index", required = True,
        help = "[Required] Similarity index file in CSV format.")
    argv.add_argument("--max", "-n", type = int, dest = "max", required = False, default = 15,
        help = "[Optional] Maximum number of similar images.")
    argv.add_argument("--mask", "-a", type = str, dest = "mask", required = False, default = "4-mask.png",
        help = "[Optional] Lung segmentation mask.")
    argv.add_argument("--model", "-m", type = str, dest = "model", required = False, default = "trained_model.hdf5",
        help = "[Optional] Lung segmentation model.")

    sys.exit(not main(argv.parse_args()))
