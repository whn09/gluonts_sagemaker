# gluonts_sagemaker

## Local Test

`python train.py --train data --test data --model-dir model --freq 1D --prediction-length 5 --context-length 50 --epochs 2 --algo-name DeepAR`

`python train.py --train data --test data --model-dir model --freq 1D --prediction-length 5 --context-length 50 --epochs 2 --algo-name LSTNet`

`python train.py --train data --test data --model-dir model --freq 1D --prediction-length 5 --context-length 50 --algo-name ARIMA` (R is supported now!!!)

`python train.py --train data --test data --model-dir model --freq 1D --prediction-length 5 --context-length 50 --algo-name Prophet` (Predictor is supported now!!!)

## Build Docker

`./build_and_push.sh gluonts_sagemaker`

## BYOC

gluonts_byoc.ipynb

## BYOS

gluonts_byos.ipynb