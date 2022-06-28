from __future__ import print_function

import argparse
import os
import logging

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
from gluonts.model.renewal import DeepRenewalProcessEstimator
from gluonts.model.rotbaum import TreeEstimator, TreePredictor
from gluonts.model.san import SelfAttentionEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster, Seq2SeqEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.model.tpp import DeepTPPEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.wavenet import WaveNetEstimator

from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.trainer import Trainer
from gluonts.mx.block.encoder import *

from gluonts.model.predictor import Predictor

from gluonts.model.naive_2 import Naive2Predictor
from gluonts.model.npts import NPTSPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator

# logging.basicConfig(level=logging.DEBUG)


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

def parse_data(dataset, use_log1p=False):
    data = []
    for t in dataset:
        if use_log1p:
            datai = {FieldName.TARGET: np.log1p(t['target']), FieldName.START: t['start']}
        else:
            datai = {FieldName.TARGET: t['target'], FieldName.START: t['start']}
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        if 'past_dynamic_feat' in t:
            datai[FieldName.PAST_FEAT_DYNAMIC_REAL] = t['past_dynamic_feat']
        data.append(datai)
    return data

def train(args):
    freq = args.freq.replace('"', '')
    prediction_length = args.prediction_length
    context_length = args.context_length
    use_feat_dynamic_real = args.use_feat_dynamic_real
    use_past_feat_dynamic_real = args.use_past_feat_dynamic_real
    use_feat_static_cat = args.use_feat_static_cat
    use_log1p = args.use_log1p
    quantiles = [0.1, 0.5, 0.9]  # TODO add to args
    
    print('freq:', freq)
    print('prediction_length:', prediction_length)
    print('context_length:', context_length)
    print('use_feat_dynamic_real:', use_feat_dynamic_real)
    print('use_past_feat_dynamic_real:', use_past_feat_dynamic_real)
    print('use_feat_static_cat:', use_feat_static_cat)
    print('use_log1p:', use_log1p)
    print('quantiles:', quantiles)
    
    batch_size = args.batch_size
    print('batch_size:', batch_size)

    train = load_json(os.path.join(args.train, 'train_'+freq+'.json'))
    test = load_json(os.path.join(args.test, 'test_'+freq+'.json'))
    
    num_timeseries = len(train)
    print('num_timeseries:', num_timeseries)
    
    algo_name = args.algo_name.replace('"', '')
    print('algo_name:', algo_name)
    
    is_multivariate = False
    if algo_name in ['DeepVAR', 'GPVAR', 'LSTNet']:
        is_multivariate = True
    print('is_multivariate:', is_multivariate)
    if is_multivariate:
#             grouper_train = MultivariateGrouper(max_target_dim=num_timeseries)
#             train_ds_multi = grouper_train(train_ds)
#             test_ds_multi = grouper_train(test_ds)
#             print('train_ds_multi:', len(train_ds_multi), next(iter(train_ds_multi)), next(iter(train_ds_multi))['target'].shape)
#             print('test_ds_multi:', len(test_ds_multi), next(iter(test_ds_multi)), next(iter(test_ds_multi))['target'].shape)
        target_dim = 1
        for i in range(len(train)):
            if len(np.array(train[i]['target']).shape) == 1:
                train[i]['target'] = [train[i]['target']]
            else:
                target_dim = len(train[i]['target'])
        for i in range(len(test)):
            if len(np.array(test[i]['target']).shape) == 1:
                test[i]['target'] = [test[i]['target']]
        print('target_dim:', target_dim)
        train_ds_multi = ListDataset(parse_data(train, use_log1p=use_log1p), freq=freq, one_dim_target=False)
        test_ds_multi = ListDataset(parse_data(test, use_log1p=use_log1p), freq=freq, one_dim_target=False)
        print('train_ds_multi:', len(train_ds_multi), next(iter(train_ds_multi)), next(iter(train_ds_multi))['target'].shape)
        print('test_ds_multi:', len(test_ds_multi), next(iter(test_ds_multi)), next(iter(test_ds_multi))['target'].shape)
    else:
        train_ds = ListDataset(parse_data(train, use_log1p=use_log1p), freq=freq)
        test_ds = ListDataset(parse_data(test, use_log1p=use_log1p), freq=freq)
        print('train_ds:', next(iter(train_ds)))
        print('test_ds:', next(iter(test_ds)))
        
    if algo_name in ['ARIMA', 'ETS', 'TBATS', 'THETAF', 'STLAR', 'CROSTON', 'MLP']:
        print('install ryp2')
        os.system('/opt/ml/code/install_rpy2.sh')
    
    predictor = None
    
    trainer= Trainer(ctx="cpu", 
                    epochs=args.epochs, 
                    num_batches_per_epoch=args.num_batches_per_epoch,
                    learning_rate=args.learning_rate, 
                    learning_rate_decay_factor=args.learning_rate_decay_factor,
                    patience=args.patience,
                    minimum_learning_rate=args.minimum_learning_rate,
                    clip_gradient=args.clip_gradient,
                    weight_decay=args.weight_decay,
                    init=args.init.replace('"', ''),
                    hybridize=args.hybridize)
    print('trainer:', trainer)
    
    cardinality = [1]
    if args.cardinality != '':
        cardinality = args.cardinality.replace('"', '').replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i in range(len(cardinality)):
            cardinality[i] = int(cardinality[i])
    print('cardinality:', cardinality)
    
    embedding_dimension = [min(50, (cat+1)//2) for cat in cardinality] if cardinality is not None else None
    print('embedding_dimension:', embedding_dimension)
    
    if algo_name == 'CanonicalRNN':
        estimator = CanonicalRNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            num_layers=5, 
            num_cells=50, 
            cell_type='lstm', 
            num_parallel_samples=100,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension[0],
        )
    elif algo_name == 'DeepFactor':
        estimator = DeepFactorEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension[0],
        )
    elif algo_name == 'DeepAR':
        estimator = DeepAREstimator(
            freq = freq,  # – Frequency of the data to train on and predict
            prediction_length = prediction_length,  # – Length of the prediction horizon
            trainer = trainer,  # – Trainer object to be used (default: Trainer())
            context_length = context_length,  # – Number of steps to unroll the RNN for before computing predictions (default: None, in which case context_length = prediction_length)
            num_layers = 2,  # – Number of RNN layers (default: 2)
            num_cells = 40,  # – Number of RNN cells for each layer (default: 40)
            cell_type = 'lstm',  # – Type of recurrent cells to use (available: ‘lstm’ or ‘gru’; default: ‘lstm’)
            dropoutcell_type = 'ZoneoutCell',  # – Type of dropout cells to use (available: ‘ZoneoutCell’, ‘RNNZoneoutCell’, ‘VariationalDropoutCell’ or ‘VariationalZoneoutCell’; default: ‘ZoneoutCell’)
            dropout_rate = 0.1,  # – Dropout regularization parameter (default: 0.1)
            use_feat_dynamic_real = use_feat_dynamic_real,  # – Whether to use the feat_dynamic_real field from the data (default: False)
            use_feat_static_cat = use_feat_static_cat,  # – Whether to use the feat_static_cat field from the data (default: False)
            use_feat_static_real = False,  # – Whether to use the feat_static_real field from the data (default: False)
            cardinality = cardinality,  # – Number of values of each categorical feature. This must be set if use_feat_static_cat == True (default: None)
            embedding_dimension = embedding_dimension,  # – Dimension of the embeddings for categorical features (default: [min(50, (cat+1)//2) for cat in cardinality])
        #     distr_output = StudentTOutput(),  # – Distribution to use to evaluate observations and sample predictions (default: StudentTOutput())
        #     scaling = True,  # – Whether to automatically scale the target values (default: true)
        #     lags_seq = None,  # – Indices of the lagged target values to use as inputs of the RNN (default: None, in which case these are automatically determined based on freq)
        #     time_features = None,  # – Time features to use as inputs of the RNN (default: None, in which case these are automatically determined based on freq)
        #     num_parallel_samples = 100,  # – Number of evaluation samples per time series to increase parallelism during inference. This is a model optimization that does not affect the accuracy (default: 100)
        #     imputation_method = None,  # – One of the methods from ImputationStrategy
        #     train_sampler = None,  # – Controls the sampling of windows during training.
        #     validation_sampler = None,  # – Controls the sampling of windows during validation.
        #     alpha = None,  # – The scaling coefficient of the activation regularization
        #     beta = None,  # – The scaling coefficient of the temporal activation regularization
            batch_size = batch_size,  # – The size of the batches to be used training and prediction.
        #     minimum_scale = None,  # – The minimum scale that is returned by the MeanScaler
        #     default_scale = None,  # – Default scale that is applied if the context length window is completely unobserved. If not set, the scale in this case will be the mean scale in the batch.
        #     impute_missing_values = None,  # – Whether to impute the missing values during training by using the current model parameters. Recommended if the dataset contains many missing values. However, this is a lot slower than the default mode.
        #     num_imputation_samples = None,  # – How many samples to use to impute values when impute_missing_values=True
        )
    elif algo_name == 'DeepState':
        estimator = DeepStateEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            batch_size=batch_size,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_static_cat=use_feat_static_cat,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
        )
    elif algo_name == 'DeepVAR':  # TODO only for multivariate, bug now
        estimator = DeepVAREstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            target_dim=target_dim,  # num_timeseries
        )
    elif algo_name == 'GaussianProcess':
        estimator = GaussianProcessEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            cardinality=num_timeseries,
        )
    elif algo_name == 'GPVAR':  # TODO only for multivariate, bug now
        estimator = GPVAREstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            target_dim=target_dim,
        )
    elif algo_name == 'LSTNet':
        estimator = LSTNetEstimator(  # use multi
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            num_series=target_dim,  # num_timeseries
            skip_size=4,
            ar_window=4,
            channels=72,
            trainer=trainer,
            batch_size=batch_size,
        )
    elif algo_name == 'NBEATS':
        estimator = NBEATSEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
        )
    elif algo_name == 'DeepRenewalProcess':
        estimator = DeepRenewalProcessEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            num_cells=40,
            num_layers=2,
        )
    elif algo_name == 'Tree':
        estimator = TreePredictor(
            freq = freq,
            prediction_length = prediction_length,
            context_length = context_length,
            n_ignore_last = 0,
            lead_time = 0,
            max_n_datapts = 1000000,
            min_bin_size = 100,  # Used only for "QRX" method.
            use_feat_static_real = False,
            use_feat_dynamic_cat = False,
            use_feat_dynamic_real = use_feat_dynamic_real,
#             cardinality = cardinality,
            one_hot_encode = False,
            model_params = {'eta': 0.1, 'max_depth': 6, 'silent': 0, 'nthread': -1, 'n_jobs': -1, 'gamma': 1, 'subsample': 0.9, 'min_child_weight': 1, 'colsample_bytree': 0.9, 'lambda': 1, 'booster': 'gbtree'},
            max_workers = 4,  # default: None
            method = "QRX",  # "QRX",  "QuantileRegression", "QRF"
            quantiles=quantiles,  # Used only for "QuantileRegression" method.
            model=None,
            seed=None,
        )
    elif algo_name == 'SelfAttention':
        # TODO: bug [KeyError: 'feat_static_cat']
        estimator = SelfAttentionEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
            model_dim=64,
            ffn_dim_multiplier=2,
            num_heads=4,
            num_layers=3,
            num_outputs=3,
            kernel_sizes=[3, 5, 7, 9],
            distance_encoding="dot",
            pre_layer_norm=False,
            dropout=0.1,
            temperature=1.0,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_dynamic_cat=False,
            use_feat_static_real=False,
            use_feat_static_cat=use_feat_static_cat,
        )
    elif algo_name == 'MQCNN':
        if use_feat_static_cat:
            estimator = MQCNNEstimator(
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                batch_size=batch_size,
                use_past_feat_dynamic_real=use_past_feat_dynamic_real,
                use_feat_dynamic_real=use_feat_dynamic_real,
                use_feat_static_cat=use_feat_static_cat,
                cardinality=cardinality,
                embedding_dimension=embedding_dimension,
                add_time_feature=True,
                add_age_feature=False,
                enable_encoder_dynamic_feature=True,
                enable_decoder_dynamic_feature=True,
                seed=None,
                decoder_mlp_dim_seq=None,
                channels_seq=None,
                dilation_seq=None,
                kernel_size_seq=None,
                use_residual=True,
                quantiles=quantiles,
                distr_output=None,
                scaling=None,
                scaling_decoder_dynamic_feature=False,
                num_forking=None,
                max_ts_len=None,
            )
        else:
            estimator = MQCNNEstimator(
                freq=freq,
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                batch_size=batch_size,
                use_past_feat_dynamic_real=use_past_feat_dynamic_real,
                use_feat_dynamic_real=use_feat_dynamic_real,
                use_feat_static_cat=use_feat_static_cat,
                add_time_feature=True,
                add_age_feature=False,
                enable_encoder_dynamic_feature=True,
                enable_decoder_dynamic_feature=True,
                seed=None,
                decoder_mlp_dim_seq=None,
                channels_seq=None,
                dilation_seq=None,
                kernel_size_seq=None,
                use_residual=True,
                quantiles=quantiles,
                distr_output=None,
                scaling=None,
                scaling_decoder_dynamic_feature=False,
                num_forking=None,
                max_ts_len=None,
            )
    elif algo_name == 'MQRNN':
        estimator = MQRNNEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            batch_size=batch_size,
        )
    elif algo_name == 'Seq2Seq':
        # TODO: but [ValueError: Deferred initialization failed because shape cannot be inferred. MXNetError: Error in operator rnnencoder0_rnn0_lstm0_rnn0: [06:38:24] ../src/operator/rnn.cc:68: Check failed: dshape.ndim() == 3U (2 vs. 3) : Input data should be rank-3 tensor of dim [sequence length, batch size, input size]]
        estimator = Seq2SeqEstimator(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            trainer=trainer,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension[0],
            encoder=RNNEncoder(  # MLPEncoder, RNNEncoder, RNNCovariateEncoder, HierarchicalCausalConv1DEncoder
                            mode='lstm',  # rnn_relu (RNN with relu activation), rnn_tanh, (RNN with tanh activation), lstm or gru.
                            hidden_size=4,
                            num_layers=4,
                            bidirectional=True,
                            use_static_feat=use_feat_static_cat,
                            use_dynamic_feat=use_feat_dynamic_real,),
            decoder_mlp_layer=[4],
            decoder_mlp_static_dim=4
        )
    elif algo_name == 'SimpleFeedForward':
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[40, 40],
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            trainer=trainer,
            batch_size=batch_size,
        )
    elif algo_name == 'TemporalFusionTransformer':
        estimator = TemporalFusionTransformerEstimator(
            prediction_length=prediction_length,
            context_length=context_length,
            freq=freq,
            trainer=trainer,
            batch_size=batch_size,
            hidden_dim=32, 
            variable_dim=None, 
            num_heads=4, 
            num_outputs=3, 
            num_instance_per_series=100, 
            dropout_rate=0.1, 
        #     time_features = [], 
        #     static_cardinalities = {}, 
        #     dynamic_cardinalities = {}, 
        #     static_feature_dims = {}, 
        #     dynamic_feature_dims = {}, 
        #     past_dynamic_features = []
        )
    elif algo_name == 'DeepTPP':
#         # TODO
#         estimator = DeepTPPEstimator(
#             prediction_interval_length=prediction_length,
#             context_interval_length=context_length,
#             freq=freq,
#             trainer=trainer,
#             batch_size=batch_size,
#             num_marks=len(cardinality) if cardinality is not None else 0,
#         )
        pass
    elif algo_name == 'Transformer':
        estimator = TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            batch_size=batch_size,
            cardinality=cardinality,
        )
    elif algo_name == 'WaveNet':
        # TODO: bug [MXNetError: Check failed: to.IsDefaultData(): ]
        estimator = WaveNetEstimator(
            freq=freq,
            prediction_length=prediction_length,
            trainer=trainer,
            batch_size=batch_size,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension[0],
        )
    elif algo_name == 'Naive2':
        # TODO Multiplicative seasonality is not appropriate for zero and negative values
        predictor = Naive2Predictor(freq=freq, prediction_length=prediction_length, season_length=context_length)
    elif algo_name == 'NPTS':
        predictor = NPTSPredictor(freq=freq, prediction_length=prediction_length, context_length=context_length)
    elif algo_name == 'SeasonalNaive':
        predictor = SeasonalNaivePredictor(freq=freq, prediction_length=prediction_length)
    elif algo_name == 'Prophet':
        # TODO: bug [RuntimeError: Can't get fully qualified name of locally defined object. train.<locals>.configure_model]
        def configure_model(model):
            model.add_seasonality(
                name='weekly', period=7, fourier_order=3, prior_scale=0.1
            )
            return model
        predictor = ProphetPredictor(freq=freq, prediction_length=prediction_length, init_model=configure_model)
    elif algo_name == 'ARIMA':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='arima',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'ETS':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='ets',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'TBATS':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='tbats',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'THETAF':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='thetaf',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'STLAR':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='stlar',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'CROSTON':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='croston',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    elif algo_name == 'MLP':
        predictor = RForecastPredictor(freq=freq,
                                      prediction_length=prediction_length,
                                      method_name='mlp',
                                      period=context_length,
                                      trunc_length=len(train[0]['target']))
    else:
        print('[ERROR]:', algo_name, 'not supported')
        return
    
    if predictor is None:
        if not is_multivariate:
            if algo_name == 'Tree':
                predictor = estimator.train(train_ds)
            else:
                predictor = estimator.train(train_ds, test_ds)
        else:
            predictor = estimator.train(train_ds_multi, test_ds_multi)

    if not is_multivariate:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
    else:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds_multi,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

    forecasts = list(forecast_it)
    tss = list(ts_it)
#     print(len(forecasts), len(tss))
    
    if not is_multivariate:
        evaluator = Evaluator(quantiles=quantiles)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
    # TODO How to evaluate multivariate?
    else: 
        if target_dim==1:
            evaluator = Evaluator(quantiles=quantiles)
        else:
            evaluator = MultivariateEvaluator(quantiles=quantiles)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds_multi))
    print(json.dumps(agg_metrics, indent=4))
    
    model_dir = os.path.join(args.model_dir, algo_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    predictor.serialize(Path(model_dir))


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--algo-name', type=str, default='DeepAR')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')  # os.environ['SM_CHANNEL_TRAINING']
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')  # os.environ['SM_CHANNEL_TEST']
    
    parser.add_argument('--freq', type=str, default='1H')
    parser.add_argument('--prediction-length', type=int, default=3*24)
    parser.add_argument('--context-length', type=int, default=7*24)

    parser.add_argument('--batch-size', type=int, default=32)
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num-batches-per-epoch', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--learning-rate-decay-factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--minimum-learning-rate', type=float, default=5e-5)
    parser.add_argument('--clip-gradient', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=1e-8)
    parser.add_argument('--init', type=str, default='xavier')
    parser.add_argument('--hybridize', action='store_true', default=False)  # TODO default True/False?
    
    parser.add_argument('--use-feat-dynamic-real', action='store_true', default=False)
    parser.add_argument('--use-feat-static-cat', action='store_true', default=False)
    parser.add_argument('--use-past-feat-dynamic-real', action='store_true', default=False)
    parser.add_argument('--cardinality', type=str, default='')
    
    parser.add_argument('--use-log1p', action='store_true', default=False)
    
    parsed, unknown = parser.parse_known_args() # this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg, type=str)

    return parser.parse_args()


def model_fn(model_dir):
    sub_dirs = os.listdir(model_dir)
#     print('[DEBUG] sub_dirs:', sub_dirs)
    for sub_dir in sub_dirs:
        if sub_dir in ['CanonicalRNN', 'DeepFactor', 'DeepAR', 'DeepState', 'DeepVAR', 'GaussianProcess', 'GPVAR', 'LSTNet', 'NBEATS', 'DeepRenewalProcess', 'Tree', 'SelfAttention', 'MQCNN', 'MQRNN', 'Seq2Seq', 'SimpleFeedForward', 'TemporalFusionTransformer', 'DeepTPP', 'Transformer', 'WaveNet', 'Naive2', 'NPTS', 'SeasonalNaive', 'Prophet', 'ARIMA', 'ETS', 'TBATS', 'THETAF', 'STLAR', 'CROSTON', 'MLP']:  # TODO add all algo_names
            model_dir = os.path.join(model_dir, sub_dir)
#             print('[DEBUG] algo_name:', sub_dir)
            algo_name = sub_dir
            break
    
    is_multivariate = False
    if algo_name in ['DeepVAR', 'GPVAR', 'LSTNet']:
        is_multivariate = True
    print('[DEBUG] algo_name:', algo_name)
    print('[DEBUG] is_multivariate:', is_multivariate)
        
    predictor = Predictor.deserialize(Path(model_dir))
    print('[DEBUG] model init done.')
    return (predictor, is_multivariate)


def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body

    
def predict_fn(input_data, model):
    (predictor, is_multivariate) = model
#     print('[DEBUG] input_data type:', type(input_data), input_data)
    if 'freq' in input_data:
        freq = input_data['freq']
    else:
        freq = '1D'
    if 'target_quantile' in input_data:
        target_quantile = float(input_data['target_quantile'])
    else:
        target_quantile = 0.5
    if 'use_log1p' in input_data:
        use_log1p = input_data['use_log1p']
    else:
        use_log1p = False
    if 'instances' in input_data:
        instances = input_data['instances']
    else:
        if isinstance(input_data, list):
            instances = input_data
        elif isinstance(input_data, dict):
            instances = [input_data]

    if is_multivariate:
        for i in range(len(instances)):
            if len(np.array(instances[i]['target']).shape) == 1:
                instances[i]['target'] = [instances[i]['target']]
        ds = ListDataset(parse_data(instances), freq=freq, one_dim_target=False)
    else:
        ds = ListDataset(parse_data(instances), freq=freq)
    
    inference_result = predictor.predict(ds)
    
    if use_log1p:
        result = [np.expm1(resulti.quantile(target_quantile)).tolist() for resulti in inference_result]
    else:
        result = [resulti.quantile(target_quantile).tolist() for resulti in inference_result]
        
    return result


def output_fn(prediction, content_type):
    return prediction


if __name__ == '__main__':
    # train
    args = parse_args()
    train(args)
    
    # predict
#     test_data = {"start": "2020-01-22 00:00:00", "target": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 5.0, 7.0, 7.0, 7.0, 11.0, 16.0, 21.0, 22.0, 22.0, 22.0, 24.0, 24.0, 40.0, 40.0, 74.0, 84.0, 94.0, 110.0, 110.0, 120.0, 170.0, 174.0, 237.0], "id": "Afghanistan_"}
#     test_data['target'] = test_data['target'][:-5]
#     data = {'instances': [test_data]}
#     data['freq'] = '1D'
#     data['target_quantile'] = 0.5
#     request_body = json.dumps(data)
    
#     model = model_fn('./model')
#     input_data = input_fn(request_body, 'application/json')
#     result = predict_fn(input_data, model)
#     output = output_fn(result, 'application/json')
#     print(output)
    