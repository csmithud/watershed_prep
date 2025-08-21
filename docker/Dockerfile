# Start from a core stack version
FROM jupyter/scipy-notebook:2022-01-03
#399cbb986c6b

LABEL authors="Mark S. Bartlett"
LABEL maintainer="bartlett <mark.bartlettjr@Stantec.com>"

COPY docker/stantec-ca.crt /usr/local/share/stantec-ca.crt
COPY docker/stantec-ca.crt /etc/ssl/certs/stantec-ca.crt
COPY docker/wincacerts.pem /usr/local/share/stantec-ca.crt

COPY --chown=${NB_UID}:${NB_GID} docker/requirements.txt /tmp/

RUN pip config set global.cert /usr/local/share/stantec-ca.crt/

RUN conda config --set ssl_verify /usr/local/share/stantec-ca.crt/
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/stantec-ca.crt
ENV PIP_CERT=/etc/ssl/certs/stantec-ca.crt

USER root

RUN cat /usr/local/share/stantec-ca.crt >> /etc/ssl/certs/stantec-ca.crt

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential \
    grass \
    grass-dev \
	libgdal-dev \
    python3-distutils \
    subversion \
    ca-certificates \
    curl \
    apt-transport-https \
    lsb-release gnupg \
    && apt-get autoremove \
    && apt-get clean

RUN NAME=$(gdal-config --version)
RUN apt-get update && apt-get install -y default-jdk 

RUN chmod -R a+rwx $HOME
RUN chown -R $NB_USER .



RUN chmod 644 /usr/local/share/stantec-ca.crt \
    && chmod 644 /etc/ssl/certs/stantec-ca.crt \
    && update-ca-certificates

# switch the user
USER $NB_USER

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/stantec-ca.crt
ENV PIP_CERT=/etc/ssl/certs/stantec-ca.crt


RUN conda config --set ssl_verify /usr/local/share/stantec-ca.crt

RUN pip config set global.cert /usr/local/share/stantec-ca.crt

RUN pip install -r /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN pip install pygdal==3.0.4.11

RUN pip install git+https://github.com/zarch/grass-session.git && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

#switch the user
USER root

RUN mkdir -p $HOME/grassdata \
   && grass -c epsg:2163  $HOME/grassdata/test -e \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.basin -s url=$HOME/grassdata/r.basin-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.accumulate -s url=$HOME/grassdata/r.accumulate-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.stream.snap -s url=$HOME/grassdata/r.stream.snap-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.stream.distance -s url=$HOME/grassdata/r.stream.distance-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.stream.basins -s url=$HOME/grassdata/r.stream.basins-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.hypso -s url=$HOME/grassdata/r.hypso-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.stream.stats -s url=$HOME/grassdata/r.stream.stats-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.stream.order -s url=$HOME/grassdata/r.stream.order-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.width.funct -s url=$HOME/grassdata/r.width.funct-74509.zip \
   && grass $HOME/grassdata/test/PERMANENT --exec g.extension extension=r.traveltime -s url=$HOME/grassdata/r.traveltime-74509.zip

RUN npm config set cafile /usr/local/share/ca-certificates/stantec-ca.crt --global
