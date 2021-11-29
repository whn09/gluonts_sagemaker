# -*- coding: utf-8 -*-
import sys
import json
import os
import warnings
import flask

from pathlib import Path

import numpy as np

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor

# The flask app for serving predictions
app = flask.Flask(__name__)
model_dir = '/opt/ml/model'
if not os.path.exists(model_dir):
    model_dir = 'model'
sub_dirs = os.listdir(model_dir)
for sub_dir in sub_dirs:
    if sub_dir in ['CanonicalRNN', 'DeepFactor', 'DeepAR', 'DeepState', 'DeepVAR', 'GaussianProcess', 'GPVAR', 'LSTNet', 'NBEATS', 'DeepRenewalProcess', 'Tree', 'SelfAttention', 'MQCNN', 'MQRNN', 'RNN2QR', 'Seq2Seq', 'SimpleFeedForward', 'TemporalFusionTransformer', 'DeepTPP', 'Transformer', 'WaveNet', 'Naive2', 'NPTS', 'Prophet', 'ARIMA', 'ETS', 'TBATS', 'CROSTON', 'MLP', 'SeasonalNaive']:  # TODO add all algo_names
        model_dir = os.path.join(model_dir, sub_dir)
        break
predictor = Predictor.deserialize(Path(model_dir))
print('model init done.')

def parse_data(dataset):
    data = []
    for t in dataset:
        datai = {FieldName.TARGET: t['target'], FieldName.START: t['start']}
        if 'id' in t:
            datai[FieldName.ITEM_ID] = t['id']
        if 'cat' in t:
            datai[FieldName.FEAT_STATIC_CAT] = t['cat']
        if 'dynamic_feat' in t:
            datai[FieldName.FEAT_DYNAMIC_REAL] = t['dynamic_feat']
        data.append(datai)
    return data

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    data = flask.request.data.decode('utf-8')
    data = json.loads(data)
#     print(data)
    if 'freq' in data:
        freq = data['freq']
    else:
        freq = '1H'
    if 'target_quantile' in data:
        target_quantile = float(data['target_quantile'])
    else:
        target_quantile = 0.5
    if 'use_log1p' in data:
        use_log1p = data['use_log1p']
    else:
        use_log1p = False
    if 'instances' in data:
        instances = data['instances']
    else:
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict):
            instances = [data]

    ds = ListDataset(parse_data(instances), freq=freq)

    inference_result = predictor.predict(ds)
    
    if use_log1p:
        result = [np.expm1(resulti.quantile(target_quantile)).tolist() for resulti in inference_result]
    else:
        result = [resulti.quantile(target_quantile).tolist() for resulti in inference_result]
    
    _payload = json.dumps(result, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
