#!/bin/bash

# Constants
GREEN='\033[0;32m'
NC='\033[0m'

# Setup environment
echo -e "${GREEN}Creating virtual environment${NC}"
python3 -m venv .venv
source .venv/bin/activate

echo -e "${GREEN}Installing packages${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}Installing pre-commit hooks${NC}"
pre-commit install

echo -e "${GREEN}Finished${NC}"
