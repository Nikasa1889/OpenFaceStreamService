#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--classifierModel', type=str, help="Path to classifier",
                    default=os.path.join(fileDir, 'celeb-classifier.nn4.small2.v1.pkl'))

parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)

with open(args.classifierModel, 'rb') as f:
    if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
    else:
            (le, clf) = pickle.load(f, encoding='latin1')

class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.fontThickness = 1
        self.textBaseline = 0
        
    def drawBox(self, rgbImg, bb, person, confidence):
        #Draw bounding box first
        cv2.rectangle(rgbImg,
                (bb.left(), bb.bottom()), (bb.right(), bb.top()),
                (127, 255, 212),
                4)
        #Calculate text length
        textSize = cv2.getTextSize(
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence),
                    self.fontFace, self.fontScale, self.fontThickness)[0]
        #Draw the text background
        cv2.rectangle(rgbImg,
                (bb.left(), bb.top()),
                (bb.left() + textSize[0], bb.top() -textSize[1]),
                (127, 255, 212), -1);
        #Now put the text on it
        cv2.putText(rgbImg,
                '{}-{:.2f}'.format(person.decode('utf-8'), confidence),
                (bb.left(), bb.top()-1),
                self.fontFace, self.fontScale,(0, 0, 0), self.fontThickness
                )
        return rgbImg

    def drawBoxes (self, rgbImg, boxes, persons, confidences):
        for box, person, confidence in zip(boxes, persons, confidences):
            rgbImg = self.drawBox(rgbImg, box, person, confidence)
        return rgbImg

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'])
            self.sendMessage('{"type": "PROCESSED"}')
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def processFrame(self, dataURL):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        #TODO: Test this!
        #rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        #rgbFrame[:, :, 0] = buf[:, :, 2]
        #rgbFrame[:, :, 1] = buf[:, :, 1]
        #rgbFrame[:, :, 2] = buf[:, :, 0]
        rgbFrame = buf
        annotatedFrame = np.copy(buf)

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        persons = []
        confidences = []
        boxes = []
        # bbs = align.getAllFaceBoundingBoxes(rgbFrame)
        bb = align.getLargestFaceBoundingBox(rgbFrame)
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            # print(len(bbs))
            landmarks = align.findLandmarks(rgbFrame, bb)
            alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                      landmarks=landmarks,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                continue
            rep = net.forward(alignedFace)
            # print(rep)
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            
            persons.append(person)
            confidences.append(confidence)
            boxes = boxes.append(bb)
        
        annotatedFrame = self.drawBoxes(annotatedFrame, boxes, persons, confidences)
        msg = {
            "type": "PERSONS",
            "identities": persons
        }
        self.sendMessage(json.dumps(msg))

        plt.figure()
        plt.imshow(annotatedFrame)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
            urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        plt.close()
        self.sendMessage(json.dumps(msg))

def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
