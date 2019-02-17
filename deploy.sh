#!/bin/bash
source activate tensorflow
mkdir -p server_log
gunicorn --workers 3 run_demo_server:app --bind 0.0.0.0:8769 --timeout 120

# 	--error-logfile server_log/error.log \
# 	--access-logfile server_log/access.log
