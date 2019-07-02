#!/bin/bash
mkdir -p server_log
GUNICORN_CMD_ARGS="-w 3 -b 0.0.0.0:8769 -t 120 \
--error-logfile server_log/error.log \
--access-logfile server_log/access.log" gunicorn run_demo_server:app
