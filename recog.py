# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 00:00:26 2018

@author: Pranay Anand
"""

from sklearn.svm import SVC
from scipy import misc
from sklearn import datasets

digits = datasets.load_digits()


features = digits.data
labels = digits.target

classifier = SVC(gamma = 0.0001)
classifier.fit(features, labels)



image = misc.imread("image.jpg")
image = misc.imresize(image, (8,8))
image = image.astype(features.dtype)
image = misc.bytescale(image,high = 16, low = 0 )
test = []
for i in image:
    for j in i:
        test.append((sum(j)/3))

print(classifier.predict([test]))