#!/bin/bash

python -m install --upgrade pip
pip install -r requirements.txt
pip install pymongo[srv]
pip install gdown
gdown --id 1miIQIphVRw2BF8PwOYQFV5sbH9ecbbE1
tar -xf IEMOCAP_mp3.tar.gz
mv .env_ .env
