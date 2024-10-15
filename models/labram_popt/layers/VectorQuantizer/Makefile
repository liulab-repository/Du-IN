#!/bin/bash

all: test

test:
	python KmeansVectorQuantizer.py
	python GumbelVectorQuantizer.py
	python LaBraMVectorQuantizer.py
	python JukeBoxVectorQuantizer.py

clean:
	rm -rf ./__pycache__

