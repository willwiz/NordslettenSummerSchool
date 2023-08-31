.PHONY: build

all: build

build:
	ifeq ($(VIRTUAL_ENV),)
		@echo "This should be installed in a python virtual environment"
	else
		pip install -e .
	endif
