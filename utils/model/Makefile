#!/bin/bash

all: test

test:
	python tf.py
	cd ./torch && make test

clean:
	rm -rf ./__pycache__
	cd ./torch && make clean

