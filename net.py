# coding: utf-8

#
# This is based on chainer examples wavenet net.py in <https://github.com/chainer/chainer/tree/master/examples/wavenet>
# Please see LICENSE-chainer.txt in the docs folder regarding to Chainer license.
#
# Change:
#  Date: Dec. 2018
#        add loss function softmax_index_weighted_sum in class wavenet
#
# Check version
#  Python 3.6.4
#  Chainer 6.0.0
#  numpy 1.14.0 


import chainer
import chainer.functions as F
import chainer.links as L
import numpy

from modules import ResidualNet


class UpsampleNet(chainer.ChainList):
    def __init__(self, out_layers, r_channels,
                 channels=[128, 128], upscale_factors=[16, 16]):
        super(UpsampleNet, self).__init__()
        for channel, factor in zip(channels, upscale_factors):
            self.add_link(L.Deconvolution2D(
                None, channel, (factor, 1), stride=(factor, 1), pad=0))
        for i in range(out_layers):
            self.add_link(L.Convolution2D(None, 2 * r_channels, 1))
        self.n_deconvolutions = len(channels)

    def __call__(self, x):
        conditions = []
        for i, link in enumerate(self.children()):
            if i < self.n_deconvolutions:
                x = F.relu(link(x))
            else:
                conditions.append(link(x))
        return F.stack(conditions)


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, a_channels, r_channels, s_channels,
                 use_embed_tanh):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.embed = L.Convolution2D(
                a_channels, r_channels, (2, 1), pad=(1, 0), nobias=True)
            self.resnet = ResidualNet(
                n_loop, n_layer, 2, r_channels, 2 * r_channels, s_channels)
            self.proj1 = L.Convolution2D(
                s_channels, s_channels, 1, nobias=True)
            self.proj2 = L.Convolution2D(
                s_channels, a_channels, 1, nobias=True)
        self.a_channels = a_channels
        self.s_channels = s_channels
        self.use_embed_tanh = use_embed_tanh
    
    def __call__(self, x, condition, w, generating=False):  # add w as weighted matrix
        length = x.shape[2]
        x = self.embed(x)
        x = x[:, :, :length, :]  # crop
        if self.use_embed_tanh:
            x = F.tanh(x)
        z = F.relu(self.resnet(x, condition))
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        
        self.w = w # keep weighted matrix data
        
        return y

    def initialize(self, n):
        self.resnet.initialize(n)

        self.embed.pad = (0, 0)
        self.embed_queue = chainer.Variable(
            self.xp.zeros((n, self.a_channels, 2, 1), dtype=self.xp.float32))

        self.proj1_queue = chainer.Variable(self.xp.zeros(
            (n, self.s_channels, 1, 1), dtype=self.xp.float32))

        self.proj2_queue3 = chainer.Variable(self.xp.zeros(
            (n, self.s_channels, 1, 1), dtype=self.xp.float32))

    def generate(self, x, condition):
        self.embed_queue = F.concat((self.embed_queue[:, :, 1:], x), axis=2)
        x = self.embed(self.embed_queue)
        if self.use_embed_tanh:
            x = F.tanh(x)
        x = F.relu(self.resnet.generate(x, condition))

        self.proj1_queue = F.concat((self.proj1_queue[:, :, 1:], x), axis=2)
        x = F.relu(self.proj1(self.proj1_queue))

        self.proj2_queue3 = F.concat((self.proj2_queue3[:, :, 1:], x), axis=2)
        x = self.proj2(self.proj2_queue3)
        return x

    '''
    add loss function
    '''
    def softmax_index_weighted_sum(self, x, t):
        # multiply  softmax as probability  and  weight between index as distance
        # x.shape=[batch, quantize, length, 1]
        
        loss= F.matmul( F.flatten(self.w) , F.flatten(F.softmax(x, axis=1)) ) / (x.data.shape[2] * x.data.shape[0])   # mean per one sample
        
        # alternate use F.tensordot that is supported at Chainer V5 and V6
        #loss= F.mean(F.tensordot(F.softmax(x, axis=1) , self.w, axes=((1),(1)))) 
        
        return loss
    
    def lossfun(self, x, t):
        # This is ordinary loss function 
        return F.softmax_cross_entropy(x, t)
    '''
    end of add loss function
    '''

class EncoderDecoderModel(chainer.Chain):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder

    def __call__(self, x, condition, w):   # add w as weighted matrix
        encoded_condition = self.encoder(condition)
        y = self.decoder(x, encoded_condition, w)
        return y

