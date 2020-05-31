FROM python:3.7.3

WORKDIR /WebApplication

ADD . /WebApplication

RUN pip install --upgrade pip

RUN  pip install -r requirements.txt


EXPOSE 70

ENV NAME World

CMD ["python", "app.py"]




