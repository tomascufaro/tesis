#!/bin/bash


packagesNeeded='gdown'
if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache $packagesNeeded
elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install $packagesNeeded
elif [ -x "$(command -v dnf)" ];     then sudo dnf install $packagesNeeded
elif [ -x "$(command -v zypper)" ];  then sudo zypper install $packagesNeeded
else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install: $packagesNeeded">&2; fi

pip install --upgrade pip
pip install pymongo[srv]
pip install notebook
pip install jupyterlab
pip install gdown
pip install pandas
pip install audiomentations
pip install datasets
pip install ipython
pip install Keras
pip install librosa
pip install matplotlib
pip install numpy
pip install packaging
pip install pymongo
pip install python-dotenv
pip install scikit_learn
pip install scipy
pip install torch
pip install torchaudio
pip install tqdm
pip install transformers



gdown --id 1miIQIphVRw2BF8PwOYQFV5sbH9ecbbE1
tar -xf IEMOCAP_mp3.tar.gz
rm IEMOCAP_mp3.tar.gz
mv ./.env_ ./.env
