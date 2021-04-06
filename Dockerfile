FROM i386/ubuntu:18.04
LABEL maintainer="Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Install Prereqisite Utils
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf
RUN echo "nameserver 8.8.4.4" > /etc/resolv.conf
RUN apt-get update --fix-missing
RUN apt-get install -y apt-utils git wget make vim tar unzip python3 python3-pip python3-h5py
RUN pip3 install pandas tqdm

# Download wDIAL
RUN git clone https://github.com/kppw99/wDIAL.git w_dial


# Set Environment Variables
RUN echo "alias q='cd ..'" >> ~/.bashrc
RUN echo "alias qq='cd ../..'" >> ~/.bashrc
RUN echo "alias qqq='cd ../../..'" >> ~/.bashrc


# Unzip raw dataset
WORKDIR /w_dial/data/trip
RUN unzip driver_A.zip
RUN unzip driver_B.zip
RUN unzip driver_F.zip
RUN unzip driver_G.zip
RUN unzip driver_I.zip
RUN rm -rf driver_A.zip driver_B.zip driver_F.zip driver_G.zip driver_I.zip


# Make csv files
WORKDIR /w_dial/source/
RUN python3 -c "from preprocessing import *; hdf5_to_csv()"

WORKDIR /data/driver_A/
RUN cp -rf /w_dial/data/trip/driver_A/*.csv .
WORKDIR /data/driver_B/
RUN cp -rf /w_dial/data/trip/driver_B/*.csv .
WORKDIR /data/driver_F/
RUN cp -rf /w_dial/data/trip/driver_F/*.csv .
WORKDIR /data/driver_G/
RUN cp -rf /w_dial/data/trip/driver_G/*.csv .
WORKDIR /data/driver_I/
RUN cp -rf /w_dial/data/trip/driver_I/*.csv .

WORKDIR /w_dial/source/
RUN python3 -c "from preprocessing import *; get_train_dataset(save='/data/train.csv')" 
RUN python3 -c "from preprocessing import *; get_test_dataset(save='/data/test.csv')" 

WORKDIR /w_dial/source/
RUN python3 -c "from svm import *; do_svm('/data/train.csv', '/data/test.csv')"

# docker build -t base/wdial:latest . -f Dockerfile
