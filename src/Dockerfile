FROM nvcr.io/nvidia/pytorch:24.05-py3

WORKDIR /app

RUN mkdir /app/local_data

COPY requirements.txt .

RUN pip install --upgrade pip
# RUN pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install -r requirements.txt

COPY ./ . 

RUN useradd -m -u 8877 nonroot
RUN chown -R 8877:8877 /app
RUN chown -R 8877:8877 /app/local_data
USER 8877

ENTRYPOINT [ "python","main.py" ]