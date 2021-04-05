"""
Originally adapted from
https://github.com/pianomania/DQN-pytorch/blob/master/memory.py
"""

import random
import numpy as np


class Replay(object):

    def __init__(self,
                 max_size=10000,
                 im_size=(2, ),
                 rand_generator=np.random.RandomState(0),
                 stack=1):
        """
        Memory efficient experience replay.
        Note that stack > 1 sampling does not work currently.
        :param max_size: Maximum buffer size.
        :param im_size: Size of a single frame/observation.
        :param stack: Frame stacking size.
        """

        # normally store np.uint8 for images, but since we don't use images, use floats
        self.s = np.zeros((max_size, *im_size), dtype=np.float32)
        self.r = np.zeros(max_size, dtype=np.float32)
        self.a = np.zeros(max_size, dtype=np.int32)
        self.done = np.array([True] * max_size)

        self.max_size = max_size

        assert stack <= 1, "Stacking of > 1 has not been implemented yet."
        self.stack = stack
        self.rand_generator = rand_generator
        self._cursor = 0
        self._filled = False
        self.eligible_idxes = None

    def start(self, state):
        self.s[self._cursor] = state

    def put(self, action, reward, done, next_state):

        self.s[(self._cursor + 1) % (self.max_size - 1)] = next_state
        self.a[self._cursor] = action
        self.r[self._cursor] = reward
        self.done[self._cursor] = done

        self._cursor = (self._cursor + 1) % (self.max_size - 1)

        if self._cursor == 0:
            # This means we're filled
            self.eligible_idxes = list(range(self.max_size))
            del self.eligible_idxes[self._cursor]
            self._filled = True
        elif not self._filled:
            self.eligible_idxes = list(range(self._cursor))

    def sample(self, bs=64):
        assert self.eligible_idxes is not None and len(self.eligible_idxes) > bs

        sample_idx = self.rand_generator.choice(self.eligible_idxes, size=bs)
        s = self.s[sample_idx]
        a = self.a[sample_idx]
        r = self.r[sample_idx]
        ss = self.s[(sample_idx + 1) % self.max_size]
        done = self.done[sample_idx]

        return s, a, r, ss, done
