#!/bin/bash

python3 test.py --config configs/test/test-div2k-2.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-3.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-4.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-6.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-8.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-12.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-18.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-24.yaml --model save/$1/epoch-last.pth
python3 test.py --config configs/test/test-div2k-30.yaml --model save/$1/epoch-last.pth

python3 test.py --config configs/test/test-set5-2.yaml --model save/$1/epoch-last.pth --name set5
python3 test.py --config configs/test/test-set5-3.yaml --model save/$1/epoch-last.pth --name set5
python3 test.py --config configs/test/test-set5-4.yaml --model save/$1/epoch-last.pth --name set5
python3 test.py --config configs/test/test-set5-6.yaml --model save/$1/epoch-last.pth --name set5
python3 test.py --config configs/test/test-set5-8.yaml --model save/$1/epoch-last.pth --name set5
python3 test.py --config configs/test/test-set14-2.yaml --model save/$1/epoch-last.pth --name set14
python3 test.py --config configs/test/test-set14-3.yaml --model save/$1/epoch-last.pth --name set14
python3 test.py --config configs/test/test-set14-4.yaml --model save/$1/epoch-last.pth --name set14
python3 test.py --config configs/test/test-set14-6.yaml --model save/$1/epoch-last.pth --name set14
python3 test.py --config configs/test/test-set14-8.yaml --model save/$1/epoch-last.pth --name set14
python3 test.py --config configs/test/test-b100-2.yaml --model save/$1/epoch-last.pth --name b100
python3 test.py --config configs/test/test-b100-3.yaml --model save/$1/epoch-last.pth --name b100
python3 test.py --config configs/test/test-b100-4.yaml --model save/$1/epoch-last.pth --name b100
python3 test.py --config configs/test/test-b100-6.yaml --model save/$1/epoch-last.pth --name b100
python3 test.py --config configs/test/test-b100-8.yaml --model save/$1/epoch-last.pth --name b100
python3 test.py --config configs/test/test-urban100-2.yaml --model save/$1/epoch-last.pth --name urban100
python3 test.py --config configs/test/test-urban100-3.yaml --model save/$1/epoch-last.pth --name urban100
python3 test.py --config configs/test/test-urban100-4.yaml --model save/$1/epoch-last.pth --name urban100
python3 test.py --config configs/test/test-urban100-6.yaml --model save/$1/epoch-last.pth --name urban100
python3 test.py --config configs/test/test-urban100-8.yaml --model save/$1/epoch-last.pth --name urban100