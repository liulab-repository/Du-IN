#!/bin/bash

all: test

test:
	python DotDict.py
	cd ./model && make test
	cd ./Paths && make test
	cd ./data && make test

clean:
	rm -rf ./__pycache__
	cd ./model && make clean
	cd ./Paths && make clean
	cd ./data && make clean

