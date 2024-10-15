#!/bin/bash

all: test

test:
	cd ./wav2spec && make test
	python mask.py
	python run_mae.py
	python run_cls.py

clean:
	rm -rf ./__pycache__
	cd ./wav2spec && make clean

