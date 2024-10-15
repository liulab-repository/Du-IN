#!/bin/bash

all: test

test:
	python data.py
	cd ./seeg && make test

clean:
	rm -rf ./__pycache__
	cd ./seeg && make clean

