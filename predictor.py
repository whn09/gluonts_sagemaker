# -*- coding: utf-8 -*-
import sys
import json
import os
import warnings
import flask

from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.predictor import Predictor

# The flask app for serving predictions
app = flask.Flask(__name__)
model_dir = '/opt/ml/model'
# model_dir = 'model'
sub_dirs = os.listdir(model_dir)
for sub_dir in sub_dirs:
    if sub_dir in ['CanonicalRNN', 'DeepFactor', 'DeepAR', ]:  # TODO add all algo_names
        model_dir = os.path.join(model_dir, sub_dir)
        break
predictor = Predictor.deserialize(Path(model_dir))

freq = '1H'
prediction_length = 3*24
context_length = 7*24
target_quantile = 0.5

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

    ds = ListDataset([{FieldName.TARGET: t['target'][:-prediction_length], FieldName.FEAT_STATIC_CAT: t['cat'], FieldName.FEAT_DYNAMIC_REAL: t['dynamic_feat'], FieldName.START: t['start'], FieldName.ITEM_ID: t['id']} for t in data], freq=freq)

    inference_result = predictor.predict(ds)
    
    result = [resulti.quantile(target_quantile).tolist() for resulti in inference_result]
    
    _payload = json.dumps(result, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
