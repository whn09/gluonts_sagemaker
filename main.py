from __future__ import print_function

import argparse
import logging
import os

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import json
import time

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas

from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.seq2seq._forking_estimator import ForkingSeq2SeqEstimator
from gluonts.block.decoder import Seq2SeqDecoder
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.block.quantile_output import QuantileOutput
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer
from gluonts.model.predictor import Predictor


logging.basicConfig(level=logging.DEBUG)

def load_json(filename):
    data = []
    with open(filename, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            datai = json.loads(line)
            data.append(datai)
    return data

# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(args):
    freq = '1H'
    prediction_length = 3*24
    context_length = 7*24

    train = load_json(os.path.join(args.train, 'train_'+freq+'.json'))
    test = load_json(os.path.join(args.train, 'test_'+freq+'.json'))
    predict = load_json(os.path.join(args.train, 'predict_'+freq+'.json'))
    print(len(train[0]['target']), len(test[0]['target']), len(predict[0]['target']))
    
    num_timeseries = len(train)
    print(num_timeseries)
    
    predict_list = []
    for t in predict:
        if len(t['target'])>=prediction_length:
            predict_list.append({FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']})
            
    train_ds = ListDataset([{FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']} for t in train], freq=freq)
    test_ds = ListDataset([{FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']} for t in test], freq=freq)
    predict_ds = ListDataset(predict_list, freq=freq)  
    
    train_entry = next(iter(train_ds))    
    test_entry = next(iter(test_ds))
    predict_entry = next(iter(predict_ds))
    
#     estimator = CanonicalRNNEstimator(
#         freq=freq,
#         prediction_length=prediction_length,
#         context_length=context_length,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        ),
#     )
    
#     estimator = DeepFactorEstimator(
#         freq=freq,
#         prediction_length=prediction_length,
#         context_length=context_length,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        ),
#     )
    
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        trainer=Trainer(ctx="cpu",
                        epochs=200,
                        learning_rate=1e-3,
                        batch_size=32,
                        num_batches_per_epoch=100
                       ),
        use_feat_dynamic_real=True,  # True
        use_feat_static_cat=True,  # True
    #     cardinality=[61]
        cardinality=[17]
    )
    
#     estimator = DeepStateEstimator(
#         freq=freq,
#         prediction_length=prediction_length,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        ),
#         use_feat_dynamic_real=True,  # True
#         use_feat_static_cat=True,  # True
#     #     cardinality=[61]
#         cardinality=[17]
#     )
    
#     # BUG
#     estimator = DeepVAREstimator(
#         freq=freq,
#         prediction_length=prediction_length,
#         context_length=context_length,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        ),
#         target_dim=1
#     )
    
#     estimator = SimpleFeedForwardEstimator(
#         num_hidden_dimensions=[40, 40],
#         prediction_length=prediction_length,
#         context_length=context_length,
#         freq=freq,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        )
#     )
        
#     # BUG: how to initialize Seq2SeqEncoder/Seq2SeqDecoder
# # estimator = ForkingSeq2SeqEstimator(
# #     encoder=Seq2SeqEncoder(),
# #     decoder=Seq2SeqDecoder(),
# #     quantile_output=QuantileOutput([0.1, 0.5, 0.9]),
# #     freq=freq,
# #     prediction_length=prediction_length,
# #     context_length=context_length,
# #     trainer=Trainer(ctx="cpu",
# #                     epochs=5,
# #                     learning_rate=1e-3,
# #                     num_batches_per_epoch=100
# #                    )
# # )
    
#     # BUG
#     estimator = WaveNetEstimator(
#         freq=freq,
#         prediction_length=prediction_length,
#         trainer=Trainer(ctx="cpu",
#                         epochs=200,
#                         learning_rate=1e-3,
#                         batch_size=32,
#                         num_batches_per_epoch=100
#                        )
#     #     cardinality=[61]
#         cardinality=[17]
#     )
    
    predictor1 = estimator.train(train_ds)
    
    predictor1.serialize(Path("gluonts_model/deepar/"))


def save(net, model_dir):
    # save the model
    net.export('%s/model'% model_dir)


def define_network():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    return net


def input_transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255.
    return data, label


def get_train_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=True, transform=input_transformer),
        batch_size=batch_size, shuffle=True, last_batch='rollover')


def get_val_data(data_dir, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.MNIST(data_dir, train=False, transform=input_transformer),
        batch_size=batch_size, shuffle=False)


def test(ctx, net, val_data):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])
    return metric.get()


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    predictor1 = Predictor.deserialize(Path("gluonts_model/deepar/"))
#     net = gluon.SymbolBlock.imports(
#         '%s/model-symbol.json' % model_dir,
#         ['data'],
#         '%s/model-0000.params' % model_dir,
#     )
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--log-interval', type=float, default=100)

    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/training')  # os.environ['SM_CHANNEL_TRAINING']

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train(args)
