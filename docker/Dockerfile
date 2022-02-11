# syntax=docker/dockerfile:1


FROM python:3.8-slim-buster
WORKDIR /app
COPY . .

RUN apt-get update 
RUN apt-get upgrade
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN pip install ipywidgets 
RUN pip install jupyterlab_widgets

RUN curl -fsSL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs
RUN yes| LC_ALL=en_US.UTF-8 apt-get install npm
RUN jupyter labextension install jupyterlab-plotly@4.14.3
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
RUN chmod +x install.sh
RUN ./install.sh


# RUN yes| LC_ALL=en_US.UTF-8 apt-get install curl
# RUN curl -sL https://deb.nodesource.com/setup_14.x | -E bash -


CMD ["/bin/bash"]

ENV NB_PREFIX /
CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]

