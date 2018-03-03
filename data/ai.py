from pynput.keyboard import Key, Controller
import pygame as pg

import numpy as np

import random
from collections import deque

import json

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
import keras.backend as K
K.set_image_data_format('channels_last')

import tensorflow as tf

keyboard = Controller()

RESUME = False
TRAIN = True
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1

GAMMA = 0.99
BATCH_SIZE = 16
REPLAY_MEMORY = 50000
EXPLORE = 30000000
OBSERVATION_TIMESTEP = 100
FRAME_PER_ACTION = 4
LEARNING_RATE = 1e-4
NUM_ACTIONS = 3
        
class AI:
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()
        self.model = self.build_model()
        self.epsilon = INITIAL_EPSILON

        x_t = skimage.color.rgb2gray(pg.surfarray.array3d(pg.display.get_surface()))
        x_t = skimage.transform.resize(x_t, (80, 80))
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) 
        self.s_t = self.s_t.reshape(
            1, 
            self.s_t.shape[0], 
            self.s_t.shape[1], 
            self.s_t.shape[2]
        )

        
        # Memory
        self.D = deque()

        self.t = 0

        if RESUME:
            self.model.load_weights("model.h5")
            adam = Adam(lr = LEARNING_RATE)
            self.model.compile(loss='mse', optimizer=adam)

    def build_model(self):
        X_input = Input((80, 80, 4))

        X = Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=((80, 80, 4)))(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Flatten()(X)

        X = Dense(128)(X)
        X = Activation('relu')(X)

        X = Dense(NUM_ACTIONS)(X)

        model = Model(inputs=X_input, outputs=X)

        adam_optimizer = Adam(lr=LEARNING_RATE)
        model.compile(optimizer=adam_optimizer, loss='mse')
        return model

    def call_ai(self, state):

        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros(NUM_ACTIONS)

        if self.t % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(NUM_ACTIONS)
                a_t[action_index] = 1
            else:
                q = self.model.predict(self.s_t)
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        if self.epsilon > FINAL_EPSILON and TRAIN:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        keyboard.press('a') if a_t[0] == 1 else keyboard.release('a')
        keyboard.press(Key.right) if a_t[1] == 1 else keyboard.release(Key.right)
        keyboard.press(Key.left) if a_t[2] == 1 else keyboard.release(Key.left)

        screenshot = state['screenshot']
        
        killed_enemy = state['killed_enemy']
        mario_dead = state['mario_dead']
        mario_big = state['mario_big']
        mario_invincible = state['mario_invincible']
        mario_in_castle = state['mario_in_castle']

        if mario_in_castle:
            r_t = r_t + 1
        if mario_dead:
            r_t = r_t - 1
        if killed_enemy:
            r_t = r_t + 0.4
        if mario_big:
            r_t = r_t + 0.6
        if mario_invincible:
            r_t = r_t + 0.7
        if a_t[1] == 1:
            r_t = r_t + 0.1

        r_t = 1 if r_t > 1 else r_t
        r_t = -1 if r_t < -1 else r_t

        terminal = True if mario_in_castle or mario_dead else False

        x_t = skimage.color.rgb2gray(screenshot)
        x_t = skimage.transform.resize(x_t, (80, 80))
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
        x_t = x_t.reshape(1, x_t.shape[0], x_t.shape[1], 1)
        s_t1 = np.append(x_t, self.s_t[:, :, :, :3], axis=3)
        self.D.append((self.s_t, action_index, r_t, s_t1, terminal))
        if (len(self.D) > REPLAY_MEMORY):
            self.D.popleft()

        if self.t > OBSERVATION_TIMESTEP:
            minibatch = random.sample(self.D, BATCH_SIZE)

            inputs = np.zeros(
                (BATCH_SIZE, 
                self.s_t.shape[1],
                self.s_t.shape[2],
                self.s_t.shape[3])
            )

            targets = np.zeros((BATCH_SIZE, NUM_ACTIONS))

            for i in range(0, BATCH_SIZE):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]

                inputs[i] = state_t

                targets[i] = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1)
                print("{}: {}".format(i, Q_sa))

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss = self.model.train_on_batch(inputs, targets)
            print(loss)

        self.s_t = s_t1
        self.t = self.t + 1

        if self.t % 1000 == 0:
            print("Saving the model")
            self.model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(self.model.to_json(), outfile)

        print(
            "TIMESTEP", self.t, "/ EPSILON", self.epsilon, \
            "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX", np.max(Q_sa), "/ Loss", loss
        )
