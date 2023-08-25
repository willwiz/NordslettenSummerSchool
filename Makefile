.PHONY: all build-dep build unittest buildtest

all: clean build-dep build unittest



build-dep:
	python -m pip install -r ./requirements.txt

build:
	python ./setup.py build_ext --build-lib .

test:
	python -m unittest discover ./unittests

buildtest: build test

clean:
	rm -rf build/*
	rm -rf src/cython/build/*
	rm -f src/py/*.pyd
	rm -f src/py/*/*.pyd
	rm -f src/py/*/*/*.pyd
	rm -f src/py/*.so
	rm -f src/py/*/*.so
	rm -f src/py/*/*/*.so
