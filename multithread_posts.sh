#!/bin/bash

NUM_THREADS=$1
NUM_REQUESTS=$2
PATH_TO_IMAGE=$3
NUM_NETS=10
REMOTE_IP="127.0.0.1:8080"

echo -e "\n"
for i in $(seq 1 $NUM_REQUESTS); do array[i]=$(($RANDOM%$NUM_NETS+1)); done
echo -e "Randomized net order is: ${array[@]}\n"
parallel --progress -j${NUM_THREADS} curl -s --form "file=@${PATH_TO_IMAGE}" --form net={} http://${REMOTE_IP}/image >/dev/null ::: ${array[@]}
echo -e "\n"
curl http://$REMOTE_IP/kill
echo -e "\n"
