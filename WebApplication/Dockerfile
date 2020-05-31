FROM python:3.7.3

WORKDIR /WebApplication

ADD . /WebApplication

RUN  pip3 install -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]