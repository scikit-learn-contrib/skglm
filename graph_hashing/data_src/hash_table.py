import numpy as np
import random

size_table = 131072  # 2 ** 17
prime = 2 ** 61 - 1  # used as a bit-mask
lowones = 2 ** 32 - 1  # used as a bit-mask


def low32of64(x):
    """ low32of64 returns the rightmost 32 bits of x
    in the form of a 64bit int (with 32 0bits on the left)"""
    return x & lowones


def high32of64(x):
    """ high32of64 returns the leftmost 32 bits of x
    in the form of a 64bit int (with 32 0bits on the left)"""
    return x >> 32


def multaddmod(x, a, b):
    """ x, a and b must be uint64. multaddmod returns (ax+b)%prime
    + possibly 2*prime using prime=2^61-1 (Mersenne prime)"""
    a0 = low32of64(a) * x
    a1 = high32of64(a) * x
    c0 = a0 + (a1 << 32)
    c1 = (a0 >> 32) + a1
    c = (c0 & prime) + (c1 >> 29) + b
    return c


class HashTable:
    def __init__(self, order=10, seed=42, multiseed=False):
        self.order = order
        self.hashtable = np.zeros((3, size_table), dtype='int64')
        self.initialized = False
        self.ai = np.zeros((3, self.order), dtype='int64')
        self.multiseed = multiseed
        self.seed = seed

    def set_order_single_init(self, new_order=10):
        self.order = new_order
        self.ai = np.zeros((3, self.order), dtype='int64')
        self.initialized = True
        if not self.multiseed:
            np.random.seed(self.seed)
            random.seed(self.seed)

        # initialize self.ai
        for i in range(3):
            for j in range(self.order):
                # pick an int between 0 and 2^31-1
                self.ai[i][j] = (random.randint(0, 0x7fffffff-1) << 32) + (random.randint(0, 0x7fffffff-1)) % prime
        # initialize self.hashtable
        for i in range(3):
            for j in range(size_table):
                acc = self.ai[i][0]
                for k in range(1, self.order):
                    acc = multaddmod(j, acc, self.ai[i][k]) % prime
                self.hashtable[i][j] = acc

    def set_order_multi_init(self, new_order=10):
        self.order = new_order
        self.initialized = True
        self.ai = np.zeros((3, self.order), dtype='int64')
        seeds = [self.seed + i for i in range(8)]

        for s in seeds:
            np.random.seed(s)
            random.seed(s)
            # initialize self.ai
            for i in range(3):
                for j in range(self.order):
                    # pick an int between 0 and 2^31-1
                    self.ai[i][j] = (random.randint(0, 0x7fffffff - 1) << 32) + (
                        random.randint(0, 0x7fffffff - 1)) % prime
            # initialize self.hashtable
            for i in range(3):
                for j in range(size_table):
                    acc = self.ai[i][0]
                    for k in range(1, self.order):
                        acc = multaddmod(j, acc, self.ai[i][k]) % prime
                    self.hashtable[i][j] <<= 8
                    self.hashtable[i][j] |= (acc & 0xFF)

    def hash16(self, val):
        if not self.initialized:
            self.set_order_single_init(new_order=10)
        return self.hashtable[0][val]

    def hash32(self, val):
        upper = val >> 16
        lower = val & 0x0000FFFF
        if not self.initialized:
            self.set_order_single_init(new_order=10)
        return self.hashtable[0][upper] ^ self.hashtable[1][lower] ^ self.hashtable[2][lower+upper]

















