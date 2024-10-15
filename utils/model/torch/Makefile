#!/bin/bash

all: test

test:
	python metrics.py
	python dist_adapter.py
	python optim_factory.py

clean:
	rm -rf ./__pycache__

