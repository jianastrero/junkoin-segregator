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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import argparse
import threading
import time

import numpy as np
import tensorflow as tf
import qrcode as qr
from PIL import Image


def main():
    # database = prepare_database()
    webcam_face_recognizer()
    return


model_file = "../tf_files/retrained_graph.pb"
label_file = "../tf_files/retrained_labels.txt"
input_height = 299
input_width = 299
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"
is_processing = False


def webcam_face_recognizer():
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """

    cv2.namedWindow("WebCam Video Feed")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        # if ready_to_detect_identity:
        #     img = process_frame(img, frame, face_cascade, database)

        process_image_thread = threading.Thread(target=process_image, args=(img,))
        process_image_thread.start()

        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow("WebCam Video Feed")


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image(image,
                           input_height=299,
                           input_width=299,
                           input_mean=0,
                           input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image_reader = image
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def process_image(img):
    global is_processing
    if not is_processing:
        is_processing = True

        graph = load_graph(model_file)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        t = read_tensor_from_image(
            img,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        identity = ""
        for i in top_k:
            if identity == "":
                identity = labels[i]
            print(labels[i], results[i])

        print("-------------identified as: " + identity)

        amount = "0"

        if identity == "trash":
            amount = "1"
        elif identity == "paper":
            amount = "5"
        elif identity == "plastic":
            amount = "10"
        elif identity == "metal":
            amount = "15"

        img = qr.make(amount)

        print(type(img))
        print(img.size)

        img.save("../qr.png")

        image = Image.open("../qr.png")
        image.show()
        time.sleep(10)
        is_processing = False

main()
