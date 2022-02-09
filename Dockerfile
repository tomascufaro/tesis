# syntax=docker/dockerfile:1


FROM python:3.8-slim-buster
WORKDIR /app
COPY . .


RUN chmod +x install.sh
RUN ./install.sh

CMD ["/bin/bash"]



