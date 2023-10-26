FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG CONFIG_PATH=configs/gap9_evk_audio.sh

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY . /gap_sdk_private

RUN apt update -y && apt upgrade -y 
RUN apt-get update &&  apt-get -y install sudo
RUN cat /gap_sdk_private/requirements_apt_ubuntu_22_04.md | xargs apt install -y

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# RUN git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu.git

RUN cd /gap_sdk_private/gap_riscv_toolchain_ubuntu && ./install.sh /usr/lib/gap_riscv_toolchain

RUN pip3 install -r /gap_sdk_private/requirements.txt
RUN pip3 install -r /gap_sdk_private/doc/requirements.txt
RUN pip3 install -r /gap_sdk_private/tools/nntool/requirements.txt
RUN pip3 install -r /gap_sdk_private/utils/gapy_v2/requirements.txt

# RUN cd /gap_sdk_private && source configs/gap9_evk_audio.sh && make all && make openocd.all && cd /gap_sdk_private/doc && make html
RUN cd /gap_sdk_private && source ${CONFIG_PATH} && make all
RUN cd /gap_sdk_private && source ${CONFIG_PATH} && make openocd.all
RUN cd /gap_sdk_private && source ${CONFIG_PATH} && cd doc && make html
RUN echo "cd /gap_sdk_private && source ${CONFIG_PATH}" >> ~/.bashrc
