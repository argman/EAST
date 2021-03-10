#!/bin/bash
set -e
cd "$(dirname "$0")"
root="$(pwd)"

# install tkinter
if dpkg-query -W -f='${Status}' python3-tk | grep "ok installed" > /dev/null 2>&1
then
  echo "tkinter already installed"
else
  sudo apt-get install python3-tk -y
fi

# install unzip
if dpkg-query -W -f='${Status}' unzip | grep "ok installed" > /dev/null 2>&1
then
  echo "unzip already installed"
else
  sudo apt-get install unzip -y
fi

# install gdown
if pip3 show gdown > /dev/null 2>&1
then
  echo "gdown already installed"
else
  pip3 install gdown
fi

# download model
mkdir -p models
cd models/
modelname="east_icdar2015_resnet_v1_50_rbox"
if [ ! -d "$modelname" ]
then
  echo "Downloading $modelname"
  modelurl='https://drive.google.com/uc?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U'
  zipname="east_icdar2015_resnet_v1_50_rbox.zip"
  if [ ! -f "$zipname" ]
  then
    gdown "$modelurl"
  else
    echo "$zipname exists"
  fi
  unzip "$zipname"
  echo "Deleting $zipname" && rm "$zipname"
else
  echo "$modelname exists"
fi

# compile lanms
cd "$root/lanms"
mkdir -p build
cd build/
cmake ..
make
cp adaptor.cpython-* ..

# install python requirements
cd "$root"
pip3 install -r test_requirements.txt
