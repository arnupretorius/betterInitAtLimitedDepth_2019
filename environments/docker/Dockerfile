# ------------------------------------------------------------------------------
# Research image for effect of initialisation in deep noisy rectifier networks.
#
# Author: Ryan Eloff
# Contact: ryan.peter.eloff@gmail.com
# Date: November 2018
# ------------------------------------------------------------------------------
# Based on the PyTorch image from https://github.com/rpeloff/research-images.
# Usage with nvidia-docker:
#   $ nvidia-docker run -it --name my_container <image-build-tag> bash
# ------------------------------------------------------------------------------
FROM reloff/pytorch-base:0.4.1-py36-cuda90
LABEL author="Ryan Eloff" date="05-11-2018"
# ------------------------------------------------------------------------------
# Install requirements from requirements.txt
# ------------------------------------------------------------------------------
COPY requirements.txt /tmp
RUN python -m pip --no-cache-dir install --upgrade -r /tmp/requirements.txt \
&& rm -rf /tmp/*
# ------------------------------------------------------------------------------
# Set default command to '/bin/sh -c bash'
# ------------------------------------------------------------------------------
CMD ["bash"]
