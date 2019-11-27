sudo apt update
sudo apt-get install autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev
sudo apt install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg
sudo apt install python-pyaudio
pip install --upgrade pip
pip install --upgrade setuptools
pip install numpy pyparsing
pip install ninja  # not required but strongly recommended
sudo apt install ocaml-nox
sudo apt install z3
sudo apt install sox
cd pykaldi/tools
./install_protobuf.sh
./install_clif.sh python /usr/lib/x86_64-linux-gnu/libpython3.6m.so.1