#! /bin/bash

cd /root/ai-module/mlops-mls

gunicorn 'mlops_serving:create_app(m="router", p="'19000'", w="58")' -c "./resources/gunicorn/gunicorn.conf.py" -b :19000 -n "gunicron mls:19000" -w 58 --preload &
service ntp start

tail -f /dev/null

