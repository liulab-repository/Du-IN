#!/bin/bash

all: test

test:
	python LambdaLayer.py
	python Tokenizer.py
	python Embedding.py
	python MultiHeadAttention.py
	python FeedForward.py
	python TransformerBlock.py
	python TransformerStack.py
	python TemporalContrastiveBlock.py
	python DecoderHead.py

clean:
	rm -rf ./__pycache__

