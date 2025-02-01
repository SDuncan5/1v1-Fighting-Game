# Street Fighter III Agent

![](.github/readme_img.jpg)

Developing a deep reinforcement learning agent to play Street Fighter III.

### Authors
- Abigayle Mercer (abmercer@calpoly.edu)
- Braedan Kennedy (bkenne07@calpoly.edu)
- Sarah Duncan (sdunca07@calpoly.edu)
- Damian Dhesi (ddhesi@calpoly.edu)
- Yayun Tan (ytan15@calpoly.edu)

### Setup
```
git clone git@github.com:SDuncan5/1v1-Fighting-Game.git
cd 1v1-Fighting-Game
./setup.sh
```

### Training
```
sudo service docker start
diambra run -s 8 -r "$PWD/roms/" python training.py --cfgFile "$PWD/cfg_files/sfiii3n/sr6_128x4_das_nc.yaml"
```
