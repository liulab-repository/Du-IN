#!/bin/bash

all: test

test:
	python duin_params.py
	python brainbert_params.py
	python brant_params.py
	python cnn_gru_params.py
	python eeg_cfmr_params.py
	python dewave_params.py
	python tstcc_params.py
	python neurobert_params.py
	python labram_params.py
	python labram_popt_params.py
	cd ./poms_params && make test

clean:
	rm -rf ./__pycache__
	cd ./poms_params && make clean

