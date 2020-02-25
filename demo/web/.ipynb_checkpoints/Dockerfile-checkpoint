# kent
FROM python:3.6 as kent

WORKDIR /tmp/kent
COPY kent/requirements.txt requirements.txt
ENV PATH="/root/.local/bin:$PATH"
RUN pip install --user --requirement=requirements.txt

COPY kent/ .
RUN pip install --user .

RUN tar --create --absolute-names --file=/tmp/env.tar \
    $(find /usr/local | grep mecab) \
    /root/.local


# jupyterlab
FROM jonghwanhyeon/jupyterlab as jupyterlab

ENV PATH="/root/.local/bin:$PATH"
COPY --from=kent /tmp/env.tar /tmp/env.tar
RUN tar --extract --absolute-names --file=/tmp/env.tar && rm /tmp/env.tar


# kepco
FROM python:3.6 as kepco

ENV PATH="/root/.local/bin:$PATH"
COPY --from=kent /tmp/env.tar /tmp/env.tar
RUN tar --extract --absolute-names --file=/tmp/env.tar && rm /tmp/env.tar

WORKDIR /app/kepco
COPY kepco/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY kepco/ .
RUN python build.py kepco/corpus/kepco.json kepco/agents kepco

ENV FLASK_APP=kepco
ENV FLASK_ENV=development
CMD flask run --host=0.0.0.0 --port=80
