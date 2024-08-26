#! /bin/bash

source $HOME/.bash_profile
BASEDIR=$(dirname "$0")

date_format=$(echo "$(date +%Y)$(date +%m)$(date +%d)$(date +%H)$(date +%M)$(date +%S)")
if [[ -z $AIMODULE_PATH ]]
then
    echo "plz export AIMODULE_PATH"
    export PYTHONPATH=PYTHONPATH:$BASEDIR
    export AIMODULE_PATH=$BASEDIR
    export AIMODULE_HOME=$BASEDIR
    export AIMODULE_LOG_PATH=$BASEDIR
fi

serving_log_dir=AIMODULE_LOG_PATH/proc/serving

if [ ! -d $serving_log_dir ]
then
    mkdir -p $serving_log_dir
fi

for arg in "$@"; do
  shift
  case "$arg" in
    '--inst_type') set -- "$@" '-i';;
  *) set -- "$@" "$arg";;
  esac
done

while getopts m:p: opts; do
    case $opts in
    m) module=$OPTARG
        ;;
    p) port=$OPTARG
        ;;
    esac
done

if [[ "${module}" == "multi" ]]
then
    gunicorn_log=${serving_log_dir}/${date_format}_gunicorn.log
    source activate ai-module
    exec bash $MLOPS_SERVING_PATH/scripts/gunicorn_inst_cmd.sh start ${module} ${port} >> $gunicorn_log 2>&1
fi
