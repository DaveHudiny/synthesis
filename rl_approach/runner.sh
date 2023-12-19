python shield_v2.py --grid-model "intercept" --learning_method "SAC" --eval-episodes 10 --constants "N=6,RADIUS=2" > logs/intercept_sac_shield.log
python shield_v2.py --grid-model "intercept" --learning_method "DDQN" --eval-episodes 10 --constants "N=6,RADIUS=2" > logs/intercept_ddqn_shield.log
python shield_v2.py --grid-model "rocks" --learning_method "SAC" --eval-episodes 10 --constants "N=6" > logs/rocks_sac_shield.log
python shield_v2.py --grid-model "rocks" --learning_method "DDQN" --eval-episodes 10 --constants "N=6" > logs/rocks_ddqn_shield.log
python shield_v2.py --grid-model "refuel" --learning_method "SAC" --eval-episodes 10 --constants "N=6,ENERGY=10" > logs/refuel_sac_shield.log
python shield_v2.py --grid-model "refuel" --learning_method "DDQN" --eval-episodes 10 --constants "N=6,ENERGY=10" > logs/refuel_ddqn_shield.log
python shield_v2.py --grid-model "evade" --learning_method "SAC" --eval-episodes 10 --constants "N=6,RADIUS=2" > logs/evade_sac_shield.log
python shield_v2.py --grid-model "evade" --learning_method "DDQN" --eval-episodes 10 --constants "N=6,RADIUS=2" > logs/evade_ddqn_shield.log
python shield_v2.py --grid-model "avoid" --learning_method "SAC" --eval-episodes 10 --constants "N=6" > logs/avoid_sac_shield.log
python shield_v2.py --grid-model "avoid" --learning_method "DDQN" --eval-episodes 10 --constants "N=6" > logs/avoid_ddqn_shield.log

exit 0

python shield_v2.py --grid-model "obstacle" --learning_method "SAC" --eval-episodes 10 --constants "N=6" > logs/obstacle_sac_shield.log
python shield_v2.py --grid-model "obstacle" --learning_method "DDQN" --eval-episodes 10 --constants "N=6" > logs/obstacle_ddqn_shield.log
