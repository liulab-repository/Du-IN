#!/bin/bash

all: test

test:
	cd ./layers && make test
	python cnn_gru.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

