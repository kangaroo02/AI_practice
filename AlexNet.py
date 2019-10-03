#! /usr/bin/env python

# pip install d2l
# pip install mxnet

import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, nn
from mxnet.gluon import loss as gloss
import os
import sys
import time
