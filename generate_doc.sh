#!/bin/bash

sphinx-apidoc -M -f -o docs/source src

cd docs

make html