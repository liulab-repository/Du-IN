#!/bin/bash

all: test

test:
	python LambdaLayer.py
	python Tokenizer.py
	python Embedding.py
	python TimeEncoder.py
	python SpatialEncoder.py
	python DecoderHead.py

clean:
	rm -rf ./__pycache__

