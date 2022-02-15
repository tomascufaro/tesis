#!/bin/bash


packagesNeeded='gdown'
if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache $packagesNeeded
elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install $packagesNeeded
elif [ -x "$(command -v dnf)" ];     then sudo dnf install $packagesNeeded
elif [ -x "$(command -v zypper)" ];  then sudo zypper install $packagesNeeded
else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install: $packagesNeeded">&2; fi

packagesNeeded='python'
if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache $packagesNeeded
elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install $packagesNeeded
elif [ -x "$(command -v dnf)" ];     then sudo dnf install $packagesNeeded
elif [ -x "$(command -v zypper)" ];  then sudo zypper install $packagesNeeded
else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install: $packagesNeeded">&2; fi

packagesNeeded='libffi-dev'
if [ -x "$(command -v apk)" ];       then sudo apk add --no-cache $packagesNeeded
elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install $packagesNeeded
elif [ -x "$(command -v dnf)" ];     then sudo dnf install $packagesNeeded
elif [ -x "$(command -v zypper)" ];  then sudo zypper install $packagesNeeded
else echo "FAILED TO INSTALL PACKAGE: Package manager not found. You must manually install: $packagesNeeded">&2; fi

pip3 install --upgrade pip3
pip3 install pymongo[srv]
pip3 install notebook
pip3 install jupyterlab
pip3 install gdown
pip3 install pandas
pip3 install audiomentations
pip3 install datasets
pip3 install ipython
pip3 install Keras
pip3 install librosa
pip3 install matplotlib
pip3 install numpy
pip3 install packaging
pip3 install pymongo
pip3 install python-dotenv
pip3 install scikit_learn
pip3 install scipy
pip3 install torch
pip3 install torchaudio
pip3 install tqdm
pip3 install transformers
pip3 install praat-parselmouth


gdown --id 1miIQIphVRw2BF8PwOYQFV5sbH9ecbbE1
tar -xf IEMOCAP_mp3.tar.gz
rm IEMOCAP_mp3.tar.gz
mv ./.env_ ./.env
