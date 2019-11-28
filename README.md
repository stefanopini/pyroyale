# Pyroyale - libRoyale Python Wrapper for Pico cameras

This is a python wrapper of libroyale, the API library of [PMD Pico depth sensors](https://pmdtec.com/picofamily/) (currently Pico Flexx, Pico Maxx, and Pico Monstar).

The wrapper is inspired by [mthrok's work](https://github.com/mthrok/libroyale-python-wrapper).

The wrapper is written in C++ and is based on the version 3.12.0.44 of the libroyale library, but works up to version 3.23.0.86.

Class names are as in the original C++ wrapper while function names are lowercase, with words separated by underscores (e.g. `get_connected_cameras`).  
The file `sample_script.py` contains an example of how to use the wrapper.
