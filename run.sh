#! /bin/bash
export PATH=$PATH:/c/Python312
export PATH=$PATH:/c/poppler/Library/bin
export PATH=$PATH:/c/poppler/Library/include
export RESUME_PARSER_HOST=0.0.0.0
export RESUME_PARSER_PORT=5001

cd application
echo Parser Running at $RESUME_PARSER_HOST:$RESUME_PARSER_PORT
python server.py
