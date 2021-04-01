# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile
FROM 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-training:1.6.0-cpu-py3
# FROM 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/mxnet-inference:1.6.0-cpu-py3

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
    
COPY ./requirements.txt /opt/ml/code/

RUN pip config set global.index-url https://opentuna.cn/pypi/web/simple/
RUN pip install -r /opt/ml/code/requirements.txt

# RUN R -e 'install.packages(c("forecast"), repos="https://cloud.r-project.org", dependencies=TRUE)'
# RUN pip install 'rpy2>=2.9.*,<3.*'

ENV PATH="/opt/ml/code:${PATH}"

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY ./ /opt/ml/code
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "serve"]