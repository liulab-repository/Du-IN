#!/bin/bash

all: test

test:
	cd ./layers && make test
	python brainbert.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

