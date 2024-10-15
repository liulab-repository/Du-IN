#!/bin/bash

all: test

test:
	cd ./layers && make test
	python labram_popt.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

