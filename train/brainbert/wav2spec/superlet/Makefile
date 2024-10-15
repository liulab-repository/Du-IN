#!/bin/bash

all: test

test:
	python superlet_sp.py
	cd ./superlet_jax && make test

clean:
	rm -rf ./__pycache__
	cd ./superlet_jax && make clean

