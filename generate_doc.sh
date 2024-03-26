#!/bin/bash

sphinx-apidoc -f -o docs/source src

cd docs

make html