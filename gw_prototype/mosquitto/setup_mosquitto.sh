#!/usr/bin/bash

sudo apt update -y && sudo apt upgrade -y
sudo apt install -y mosquitto
cp mosquitto.conf.pi /etc/mosquitto/mosquitto.conf
sudo systemctl enable mosquitto.service