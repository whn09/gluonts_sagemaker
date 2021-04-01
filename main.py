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

from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster, Seq2SeqEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.wavenet import WaveNetEstimator

from gluonts.block.quantile_output import QuantileOutput
from gluonts.trainer import Trainer
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.model.predictor import Predictor

from gluonts.model.naive_2 import Naive2Predictor
from gluonts.model.npts import NPTSPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor


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
    freq = args.freq  # '1H'
    prediction_length = args.prediction_length  # 3*24
    context_length = args.context_length  # 7*24

    train = load_json(os.path.join(args.train, 'train_'+freq+'.json'))
    test = load_json(os.path.join(args.train, 'test_'+freq+'.json'))
    predict = load_json(os.path.join(args.train, 'predict_'+freq+'.json'))
    
    num_timeseries = len(train)
    
    predict_list = []
    for t in predict:
        if len(t['target'])>=prediction_length:
            predict_list.append({FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']})
            
    train_ds = ListDataset([{FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']} for t in train], freq=freq)
    test_ds = ListDataset([{FieldName.TARGET: t['target'], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']} for t in test], freq=freq)
    predict_ds = ListDataset(predict_list, freq=freq)  
    
    grouper_train = MultivariateGrouper(max_target_dim=num_timeseries)
    train_ds_multi = grouper_train(train_ds)
    test_ds_multi = grouper_train(test_ds)
    predict_ds_multi = grouper_train(predict_ds)
    
    trainer= Trainer(ctx="cpu", epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, num_batches_per_epoch=100)
    
    if args.algo_name == 'CanonicalRNN':
        estimator = CanonicalRNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
    elif args.algo_name == 'DeepFactor':
        estimator = DeepFactorEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
    elif args.algo_name == 'DeepAR':
        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            use_feat_dynamic_real=True,  # True
            use_feat_static_cat=True,  # True
        #     cardinality=[61]
            cardinality=[17]
        )
    elif args.algo_name == 'DeepState':
        estimator = DeepStateEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            use_feat_dynamic_real=True,  # True
            use_feat_static_cat=True,  # True
        #     cardinality=[61]
            cardinality=[17]
        )
    elif args.algo_name == 'DeepVAR':
        estimator = DeepVAREstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            target_dim=96
        )
    elif args.algo_name == 'GaussianProcess':
        estimator = GaussianProcessEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            #     cardinality=61
            cardinality=17
        )
    elif args.algo_name == 'GPVAR':
        estimator = GPVAREstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            target_dim=96
        )
    elif args.algo_name == 'LSTNet':
        estimator = LSTNetEstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            num_series=96,
            skip_size=4,
            ar_window=4,
            channels=72,
            trainer=trainer,
        )
    elif args.algo_name == 'NBEATS':
        estimator = NBEATSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
    elif args.algo_name == 'MQCNN':
        estimator = MQCNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
    elif args.algo_name == 'MQRNN':
        estimator = MQRNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
        )
    elif args.algo_name == 'RNN2QR':
        # # TODO
        # estimator = RNN2QRForecaster(
        #     freq=freq,
        #     prediction_length=prediction_length,
        #     context_length=context_length,
        #     trainer=trainer,
        # #     cardinality=[61]
        #     cardinality=[17],
        #     embedding_dimension=4,
        #     encoder_rnn_layer=4,
        #     encoder_rnn_num_hidden=4,
        #     decoder_mlp_layer=[4],
        #     decoder_mlp_static_dim=4
        # )
        pass
    elif args.algo_name == 'Seq2Seq':
        # # TODO
        # estimator = Seq2SeqEstimator(
        #     freq=freq,
        #     prediction_length=prediction_length,
        #     context_length=context_length,
        #     trainer=trainer,
        # #     cardinality=[61]
        #     cardinality=[17],
        #     embedding_dimension=4,
        #     encoder=Seq2SeqEncoder(),
        #     decoder_mlp_layer=[4],
        #     decoder_mlp_static_dim=4
        # )
        pass
    elif args.algo_name == 'SimpleFeedForward':
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[40, 40],
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            trainer=trainer,
        )
    elif args.algo_name == 'Transformer':
        estimator = TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
        #     cardinality=[61]
            cardinality=[17]
        )
    elif args.algo_name == 'WaveNet':
        estimator = WaveNetEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
        #     cardinality=[61]
            cardinality=[17]
        )
#     elif args.algo_name == 'Naive2':
#         # # TODO Multiplicative seasonality is not appropriate for zero and negative values
#         # predictor = Naive2Predictor(freq=freq, prediction_length=prediction_length, season_length=context_length)
#         pass
#     elif args.algo_name == 'NPTS':
#         predictor = NPTSPredictor(freq=freq, prediction_length=prediction_length, context_length=context_length)
#     elif args.algo_name == 'Prophet':
#         def configure_model(model):
#             model.add_seasonality(
#                 name='weekly', period=7, fourier_order=3, prior_scale=0.1
#             )
#             return model
#         predictor = ProphetPredictor(freq=freq, prediction_length=prediction_length, init_model=configure_model)
#     elif args.algo_name == 'ARIMA':
#         # TODO
#         predictor = RForecastPredictor(freq=freq,
#                                       prediction_length=prediction_length,
#                                       method_name='arima',  # The method from rforecast to be used one of “ets”, “arima”, “tbats” (bug), “croston” (bug), “mlp” (bug).
#                                       period=context_length,
#                                       trunc_length=len(train[0]['target']))
# #         pass
#     elif args.algo_name == 'SeasonalNaive':
#         predictor = SeasonalNaivePredictor(freq=freq, prediction_length=prediction_length)
    else:
        print('[ERROR]:', args.algo_name, 'not supported')
    
    try:
        predictor = estimator.train(train_ds)
    except:
        predictor = estimator.train(train_ds_multi)
    
    model_dir = os.path.join(args.model_dir, args.algo_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    predictor.serialize(Path(model_dir))


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--freq', type=str, default='1H')
    parser.add_argument('--prediction-length', type=int, default=3*24)
    parser.add_argument('--context-length', type=int, default=7*24)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/training')  # os.environ['SM_CHANNEL_TRAINING']
    parser.add_argument('--algo-name', type=str, default='DeepAR')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train(args)
