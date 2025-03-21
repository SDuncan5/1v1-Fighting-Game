# Street Fighter III Agent

![](.github/readme_img.jpg)

Developing a deep reinforcement learning agent to play Street Fighter III.

### Acknowledgements
This is a project for CSC 570; Aritificial Intelligence in Games, taught by Prof. Canaan at Cal Poly San Luis Obispo.

We reply heavily on the following libraries:
- [Diambra](https://github.com/diambra)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Stable-Baselines3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

### Authors
- Abigayle Mercer (abmercer@calpoly.edu)
- Braedan Kennedy (bkenne07@calpoly.edu)
- Damian Dhesi (ddhesi@calpoly.edu)
- Sarah Duncan (sdunca07@calpoly.edu)
- Yayun Tan (ytan15@calpoly.edu)

### Setup
```
git clone git@github.com:SDuncan5/1v1-Fighting-Game.git
cd 1v1-Fighting-Game
./setup.sh
```

### Training
To configure training parameters:
```
vim cfg_files/sfiii3n/sr6_128x4_das_nc.yaml
```

To start training with 8 parallel environments:
```
sudo service docker start
source .venv/bin/activate
diambra run -s 8 -r "$PWD/roms/" python training.py --cfgFile "$PWD/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
```

### Evaluating
To start evaluation of an agent:
```
sudo service docker start
source .venv/bin/activate
diambra run -r "$PWD/roms/" python evaluate.py --cfgFile "$PWD/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
```

### Tensorboard
```
source .venv/bin/activate
tensorboard --logdir results/sfiii3n/sr6_128x4_das_nc/tb
```
