FROM Python 3.7.3

WORKDIR /WebApplication

ADD . /WebApplication

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]