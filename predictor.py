# -*- coding: utf-8 -*-
import sys
import json
import os
import warnings
import flask
from gluonts.model.predictor import Predictor

# The flask app for serving predictions
app = flask.Flask(__name__)
predictor = Predictor.deserialize(os.path.join('/opt/ml/model', 'gluonts_model/deepar/'))

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
    print(data)

    inference_result = predictor.predict(data)
    
    _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')