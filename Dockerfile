# syntax=docker/dockerfile:1


FROM python:3.8-slim-buster
WORKDIR /app
COPY . .


RUN chmod +x install.sh
RUN ./install.sh

RUN pip install jupyterlab, ipwidgets, jupyterlab_widgets

RUN jupyter labextension install jupyterlab-plotly@4.14.3
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3

CMD ["/bin/bash"]

ENV NB_PREFIX /
CMD ["sh","-c", "jupyter lab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]

