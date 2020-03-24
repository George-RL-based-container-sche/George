#!/bin/sh

pathname="CPO_unified_RL2_more.py"

for VARIABLE in 0 1 2 3 4
do
python3 $pathname --batch_choice $(($VARIABLE + $1))
done

# etc.