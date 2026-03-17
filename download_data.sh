#!/bin/bash
curl -L -o data/real-or-fake-fake-jobposting-prediction.zip\
  https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction

unzip data/real-or-fake-fake-jobposting-prediction.zip -d data/

rm data/real-or-fake-fake-jobposting-prediction.zip
