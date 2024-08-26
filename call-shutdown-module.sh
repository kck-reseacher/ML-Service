#! /bin/bash
source $HOME/.bash_profile

SS_PID=$1

shutdown()
{
  SS_PID=$1
  retval=0

  SERVING_PS=$(ps -ef | awk '$3 ~ /^'"${SS_PID}"'$/ {print}' | awk '{print $2}')

  if [ -n SERVING_PS ];
  then
    for S_PID in $SERVING_PS
    do
      CC_PS=$(ps -ef | awk '$3 ~ /^'"${S_PID}"'$/ {print}')
      if [ "${CC_PS}" != "" ]; then
        # child_ps의 cc_pid
        CC_PID=$(ps -ef | awk '$3 ~ /^'"${S_PID}"'$/ {print}' | awk '{print $2}')

        # kill serving ps - child ps
#        echo >&2 "kill -9 ${CC_PID}"
        kill -9 $CC_PID
      fi
    done
    # kill serving ps
#    echo "kill -9 ${SERVING_PS}"
    kill -9 ${SERVING_PS}
  fi

  GUNICORNPROCESS=(`ps -ef | grep 'exem_aiops_anls_inst_multi' | grep 'master' | grep -v 'grep' | awk '{print $2}'`)

#  echo >&2 'gunicorn master process list' ${GUNICORNPROCESS[*]}

  for ((i=0;i<${#GUNICORNPROCESS[@]};i++))
  do
#    echo >&2 'kill gunicorn master process '${GUNICORNPROCESS[$i]}
    kill -TERM ${GUNICORNPROCESS[$i]}
#    echo >&2 'gunicorn master server TERM signal ing..'
    sleep 1
  done

  retval=1

  return "$retval"
}
shutdown $SS_PID
retval=$?

if [ "$retval" == 1 ]
then
     echo 1
else
     echo -1
fi
