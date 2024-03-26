#!/bin/bash

rm -rf docs
mkdir -p docs

python -m pydoc -w ./

mv *.html docs/

