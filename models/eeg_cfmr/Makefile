#!/bin/bash

all: test

test:
	cd ./layers && make test
	python eeg_cfmr.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

