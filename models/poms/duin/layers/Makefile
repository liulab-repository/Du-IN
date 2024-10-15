#!/bin/bash

all: test

test:
	python LambdaLayer.py
	python ResidualConnection.py
	python GradScaler.py
	python Tokenizer.py
	python Embedding.py
	python MultiHeadAttention.py
	python FeedForward.py
	python TransformerBlock.py
	python TransformerStack.py
	cd ./VectorQuantizer && make test
	python ContrastiveBlock.py
	python DecoderHead.py

clean:
	rm -rf ./__pycache__
	cd ./VectorQuantizer && make clean

