#!/bin/sh

ALPHA=$1
pathname="CPO_unified_RL1_1_more_new.py"

for VARIABLE in 0 1 2 3 4
do
python3 $pathname --alpha $ALPHA --batch_choice $(($VARIABLE+$2))
done

# etc.