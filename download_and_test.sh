#!/bin/sh

cd src

printf "\nDownloading data...\n"
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz

printf "\nRunning smoke test...\n"
pipenv run python vggish_smoke_test.py