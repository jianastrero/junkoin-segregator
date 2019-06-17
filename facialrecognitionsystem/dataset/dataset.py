#  MIT License
#
#  Copyright (c) 2019 Jian James Astrero
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#

import cv2
import os
import numpy as np

from facialrecognitionsystem.face.FaceDetector import FaceDetector
from facialrecognitionsystem.face.face_util import normalize_faces


def train_dataset():
    images = []
    labels = []
    labels_dic = {}
    paths = []
    train_directory = os.path.join(os.path.dirname(__file__), 'train') + '/'
    people = [person for person in os.listdir(train_directory)]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir(train_directory + person):
            path = train_directory + person + '/' + image
            paths.append(path)
            images.append(cv2.imread(path, 0))
            labels.append(person)

    return images, np.array(labels), labels_dic, paths


def test_dataset():
    pass


def normalize_dataset(images, paths):
    count = 0
    xml_path = os.path.join(os.path.dirname(__file__), '../haarcascade_frontalface_default.xml')
    for image, path in zip(images, paths):
        detector = FaceDetector(xml_path)
        faces_coord = detector.detect(image, True)
        faces = normalize_faces(image, faces_coord)
        for i, face in enumerate(faces):
            cv2.imwrite('normalized_' + path, faces[i])
            count += 1
    return count
