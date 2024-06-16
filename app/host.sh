#!/bin/bash

source .venv/bin/activate
echo "inital run might take a minute. please standby..."
python -m flask run --host=127.0.0.1 --port=8080