#!/usr/bin/env bash

bash -c "python noisyart_metadata.py '$1'; echo ; read -n 1 -s -p \"..Press a key to terminate..\"; echo"

