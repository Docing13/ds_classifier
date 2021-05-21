FROM python:3.9

WORKDIR /dru_final_project

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/app'

COPY . .

CMD ["python", "./app.py"]