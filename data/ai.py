from pynput.keyboard import Key, Controller
import pygame as pg

import numpy as np

import json

import skimage as skimage
from skimage.transform import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import 

from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

import tensorflow as tf

keyboard = Controller()

class AI:
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()
        self.counter = 0

    def build_model(self):
        model = Sequential()

    def publish_state(self, state):

        screenshot = state['screenshot']


        # self.counter = self.counter + 1
        # if self.counter % 10 == 0:
        #     keyboard.release('a')
        # else:
        #     keyboard.press('a')
        
        keyboard.press(Key.right)