FROM base/wdial:latest
LABEL maintainer="Sanghoon(Kevin) Jeon <kppw99@gmail.com>"


# Install library
RUN pip3 install scikit-learn


# Update github of wDIAL
WORKDIR /w_dial/
RUN git pull origin main


# do_svm()
WORKDIR /w_dial/source/
RUN python3 -c "from svm import *; do_svm('/data/train.csv', '/data/test.csv')"

# docker build -t svm/wdial:latest . -f svm.Dockerfile
