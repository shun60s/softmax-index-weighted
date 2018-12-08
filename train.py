# coding: utf-8

#
# This is based on chainer examples wavenet train.py in <https://github.com/chainer/chainer/tree/master/examples/wavenet>
# Please see LICENSE-chainer.txt in the docs folder regarding to Chainer license.
#
# Change:
#  Date: Dec. 2018
#        An experimental loss function softmax_index_weighted_sum which is declared in net.py
#        weighted matrix is computed during Preprocess output in utils.py
#        
#        add use normal Iterator (not MultiprocessIterator)
#        n_loop default from 4 to 2 as same as Chainer Colab Notebooks value
#
#
# Check version
#  Python 3.6.4
#  Chainer 6.0.0
#  numpy 1.14.0 


import argparse
import pathlib

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions


from net import EncoderDecoderModel
from net import UpsampleNet
from net import WaveNet
from utils import Preprocess
from receptive_width import show_receptive_width # add


parser = argparse.ArgumentParser(description='Chainer example: WaveNet')
parser.add_argument('--batchsize', '-b', type=int, default=5,  # !Set 3 if memory size is not enough
                    help='Numer of audio clips in each mini-batch')
parser.add_argument('--length', '-l', type=int, default=5120,  # changed to 5120=256 x 20frames
                    help='Number of samples in each audio clip')
parser.add_argument('--epoch', '-e', type=int, default=500,  # changed 
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=0,  # !set -1 if use CPU, without GPU
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-i', default='./wav100', # change dataset name
                    help='Directory of dataset')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--n_loop', type=int, default=2,  # change from 3 to 2 as same as Chainer Colab Notebooks value
                    help='Number of residual blocks')
parser.add_argument('--n_layer', type=int, default=10,
                    help='Number of layers in each residual block')
parser.add_argument('--a_channels', type=int, default=256,
                    help='Number of channels in the output layers')
parser.add_argument('--r_channels', type=int, default=64,
                    help='Number of channels in residual layers and embedding')
parser.add_argument('--s_channels', type=int, default=256,
                    help='Number of channels in the skip layers')
parser.add_argument('--use_embed_tanh', type=bool, default=True,
                    help='Use tanh after an initial 2x1 convolution')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed to split dataset into train and test')
parser.add_argument('--snapshot_interval', type=int, default=50, # !changed
                    help='Interval of snapshot')
parser.add_argument('--display_interval', type=int, default=4,   # !changed
                    help='Interval of displaying log to console')
parser.add_argument('--process', type=int, default=2,  # !set 0 if you use single device CPU
                    help='Number of parallel processes, set 0 if use normal Iterator')
parser.add_argument('--prefetch', type=int, default=8,  # ignore if --process is 0
                    help='Number of prefetch samples')
parser.add_argument('--alpha', '-a', type=float, default=1E-4,  # add
                    help='Coefficient of learning rate of optimizer Adam')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

if args.gpu >= 0:
    chainer.global_config.autotune = True
    use_gpu = True  # add

# Datasets
paths = sorted([
    str(path) for path in pathlib.Path(args.dataset).glob('*.wav')])  # change wav files path

preprocess = Preprocess(
    sr=16000, n_fft=1024, hop_length=256, n_mels=128, top_db=20,
    length=args.length, quantize=args.a_channels)
dataset = chainer.datasets.TransformDataset(paths, preprocess)
train, valid = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.9), args.seed)

# Networks
encoder = UpsampleNet(args.n_loop * args.n_layer, args.r_channels)
decoder = WaveNet(
    args.n_loop, args.n_layer,
    args.a_channels, args.r_channels, args.s_channels,
    args.use_embed_tanh)

show_receptive_width(args.n_layer, args.n_loop, residual_conv_filter_width=2, sampling_rate=16000) # add

# loss function choice
if 1:
    model = chainer.links.Classifier(EncoderDecoderModel(encoder, decoder), lossfun=decoder.softmax_index_weighted_sum)
else:
    model = chainer.links.Classifier(EncoderDecoderModel(encoder, decoder), lossfun=decoder.lossfun)

# Optimizer
optimizer = chainer.optimizers.Adam( args.alpha )
optimizer.setup(model)

# Iterators
if args.process == 0:  # use normal Iterator
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize,)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
else:  # use MultiprocessIterator
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, args.batchsize, repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)


# Updater and Trainer
updater = chainer.training.StandardUpdater(
    train_iter, optimizer, device=args.gpu)
trainer = chainer.training.Trainer(
    updater, (args.epoch, 'epoch'), out=args.out)

# Extensions
snapshot_interval = (args.snapshot_interval, 'iteration')
display_interval = (args.display_interval, 'iteration')
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
trainer.extend(extensions.LogReport(trigger=display_interval))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=display_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'iteration', file_name='loss.png', trigger=display_interval))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'iteration', file_name='accuracy.png', trigger=display_interval))
trainer.extend(extensions.ProgressBar())  # update_interval=10))

# Resume
if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# Run
trainer.run()

