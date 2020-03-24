#!/bin/sh

ps -ef | grep python

pkill screen
pkill python
pkill pyhon3
cd /home/ubuntu/atc/simulated_env
git pull

#screen -r MySessionName0


screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -dmS MySessionName3 &
screen -dmS MySessionName4 &
screen -dmS MySessionName5 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 5
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 10
" &
screen -S MySessionName3 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 15
" &
screen -S MySessionName4 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 20
" &
screen -S MySessionName5 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.1 25
"





screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -dmS MySessionName3 &
screen -dmS MySessionName4 &
screen -dmS MySessionName5 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 5
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 10
" &
screen -S MySessionName3 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 15
" &
screen -S MySessionName4 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 20
" &
screen -S MySessionName5 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.2 25
"





screen -dmS MySessionName0 &
screen -dmS MySessionName1 &
screen -dmS MySessionName2 &
screen -dmS MySessionName3 &
screen -dmS MySessionName4 &
screen -dmS MySessionName5 &
screen -S MySessionName0 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 0
" &
screen -S MySessionName1 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 5
" &
screen -S MySessionName2 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 10
" &
screen -S MySessionName3 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 15
" &
screen -S MySessionName4 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 20
" &
screen -S MySessionName5 -p 0 -X stuff "cd /home/ubuntu/atc/simulated_env;  sh cpo_compare_27_RL1_more.sh 0.9 25
"

