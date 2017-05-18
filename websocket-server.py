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
#from twisted.internet import task, defer
#from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log
from twisted.internet import reactor

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
import dlib
from FaceDetection import FaceDetection

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
# For TLS connections
#tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
#tls_key = os.path.join(fileDir, 'tls', 'server.key')

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
        #super(OpenFaceServerProtocol, self).__init__()
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")
        self.faceDetection = FaceDetection(cuda=False);
        
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
        elif msg['type'] == "FRAME_WITH_BBS":   
            self.processFrameWithBBs(msg['dataURL'], msg['bbs'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "FACE_WITH_BBS":
            self.processFaceArray(msg['face'], msg['bbs'])
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
    
    
    def getImgFromDataURL(self, dataURL):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        return np.asarray(img)
    
    def getDlibRectanglesFromBBs(self, bbs):
        dlibRectangles = []
        for bb in bbs:
            dlibRectangles.append(dlib.rectangle(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]))
        return dlibRectangles
    
    def convertImgToBase64(self, img):
        plt.figure()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='jpg')
        imgdata.seek(0)
        #NOTE: do not use urllib.quote, this cause trouble with c++ base64 decode
        #content = 'data:image/jpeg;base64,' + \
        #    urllib.quote(base64.b64encode(imgdata.buf))
        content = 'data:image/jpeg;base64,' + base64.b64encode(imgdata.buf)
        plt.close()
        return content
    
    def processFace(self, face, bbs):
        face_img = np.asarray(face)
        dlibRectangles = self.getDlibRectanglesFromBBs(bbs)
        (persons, confidences) = self.faceDetection.infer(face_img, bbs=dlibRectangles, drawBox=False)
        content = self.convertImgToBase64(face_img)
        msg = {
            "type": "ANNOTATED",
            "content": content,
            "persons": persons,
            "confidences": confidences
        }
        self.sendMessage(json.dumps(msg))
        
    def processFrameWithBBs(self, dataURL, bbs):
        rgbFrame = self.getImgFromDataURL(dataURL)
        annotatedFrame = np.copy(rgbFrame)
        dlibRectangles = self.getDlibRectanglesFromBBs(bbs)
        annotatedFrame = self.faceDetection.infer(annotatedFrame, bbs = dlibRectangles)
        
        content = self.convertImgToBase64(annotatedFrame)
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        
        self.sendMessage(json.dumps(msg))
    
    def processFrame(self, dataURL):
        rgbFrame = self.getImgFromDataURL(dataURL)
        annotatedFrame = np.copy(rgbFrame)
           
        annotatedFrame = self.faceDetection.infer(annotatedFrame)

        content = self.convertImgToBase64(annotatedFrame)
        
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        
        self.sendMessage(json.dumps(msg))

# def main(reactor):
    # log.startLogging(sys.stdout)
    # factory = WebSocketServerFactory()
    # factory.protocol = OpenFaceServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    # reactor.listenSSL(args.port, factory, ctx_factory)
    # return defer.Deferred()

if __name__ == '__main__':
    #task.react(main)
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port), debug=False)
    factory.protocol = OpenFaceServerProtocol
    reactor.listenTCP(args.port, factory)
    reactor.run()
