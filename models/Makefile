#!/bin/bash

all: test

test:
	cd ./duin && make test
	cd ./brainbert && make test
	cd ./brant && make test
	cd ./cnn_gru && make test
	cd ./eeg_cfmr && make test
	cd ./dewave && make test
	cd ./tstcc && make test
	cd ./neurobert && make test
	cd ./labram && make test
	cd ./labram_popt && make test
	cd ./poms && make test

clean:
	rm -rf ./__pycache__
	cd ./duin && make clean
	cd ./brainbert && make clean
	cd ./brant && make clean
	cd ./cnn_gru && make clean
	cd ./eeg_cfmr && make clean
	cd ./dewave && make clean
	cd ./tstcc && make clean
	cd ./neurobert && make clean
	cd ./labram && make clean
	cd ./labram_popt && make clean
	cd ./poms && make clean

