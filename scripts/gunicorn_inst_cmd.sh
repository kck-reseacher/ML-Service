#! /bin/bash


input=$1
input_module=$2
input_port=$3

echo -e "\n********************************* gunicorn server ${input} *********************************"

source $HOME/.bash_profile
  BASEDIR=$(dirname "$0")

  if [[ -z $AIMODULE_PATH ]]
  then
      echo "plz export AIMODULE_PATH"
      export PYTHONPATH=PYTHONPATH:$BASEDIR
      export AIMODULE_PATH=$BASEDIR
      export AIMODULE_HOME=$BASEDIR
  fi

start()
{
  MODULE_NAME=$1
  input_port=$2
  WORKER_CNT=$3

  source activate ai-module
  echo ${MODULE_NAME}' module '${input_port}' port,  gunicorn server' ${WORKER_CNT}' worker run..'
  cd ${MLOPS_SERVING_PATH} && gunicorn 'mlops_serving:create_app(m="'${MODULE_NAME}'", p="'${input_port}'", w="'${WORKER_CNT}'")' -c "./resources/gunicorn/gunicorn.conf.py" -b :${input_port} -n "mlops_serving:${input_port}" -w ${WORKER_CNT} --preload

}

echo '------------------------------------------------------------------------------------------'

MODULE_NAME_MULTI="router"
MULTI_WORKER_CNT=58
INPUT_PORT=$input_port

case $input in
  start) start ${MODULE_NAME_MULTI} ${INPUT_PORT} ${MULTI_WORKER_CNT}
    ;;
  *)
    echo 'start 명령어로 실행해주세요.'
    ;;
esac

echo '------------------------------------------------------------------------------------------'
