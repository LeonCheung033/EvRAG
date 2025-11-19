#!/bin/bash
# 自动生成和更新requirements.txt

echo "Generating requirements.txt using pipreqs..."
pipreqs . --force --encoding=utf8 --savepath requirements.txt

echo "Checking dependency tree..."
pipdeptree

echo "Checking for conflicts..."
pip check