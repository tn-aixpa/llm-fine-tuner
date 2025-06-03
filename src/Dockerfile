FROM nvcr.io/nvidia/pytorch:24.05-py3

RUN pip install --upgrade pip

COPY requirements.txt /code/

RUN pip install -r /code/requirements.txt

COPY ./ /code/

COPY run_training.sh /code/run_training.sh

WORKDIR /code


CMD ["sh", "run_training.sh"]


