# gluonts_sagemaker

## Local Test

`python main.py --train data --model-dir model --epochs 2 --algo-name DeepAR`

`python main.py --train data --model-dir model --epochs 2 --algo-name LSTNet`

`python main.py --train data --model-dir model --algo-name ARIMA` (R is not supported now.)

`python main.py --train data --model-dir model --algo-name Prophet` (Predictor is not supported now.)

## Build Docker

`./build_and_push.sh gluonts_sagemaker`

## BYOC

gluonts_byoc.ipynb

## BYOS

gluonts_byos.ipynb