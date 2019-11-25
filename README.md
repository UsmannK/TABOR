#TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems 

## About
This repository contains partial code implementation of the paper (https://arxiv.org/pdf/1908.01763.pdf). Currently this repo has been written to work on the GTSRB dataset with the 6 Conv + 2 MaxPooling CNN from the original paper.

## Dependencies

This codebase is written in tensorflow and tf.keras and has been tested on tensorflow 1.14 and python 3.6.8

## Getting Started

1. Clone the TABOR repository
    ```shell
    git clone https://github.com/UsmannK/TABOR.git
    ```

2. Download the training data (GTSRB signs)
    ```shell
    cd TABOR
    ./download_data.sh
    ```

3. Run the BadNet trainer:
    ```shell
    python3 tabor/train_badnet.py --train --poison-type FF --poison-loc TL --poison-size 8 --epochs 10 --display
    ```
    Currently supported options:
    
    poison type: `FF` (firefox logo) and `whitesquare`
    poison location: `TL` and `BR`: Top Left and Bottom Right
    poison size: integers

    To train without any poison, exclude the `poison-*` options

4. Run TABOR on the best model in `output/`
    ```shell
    python3 tabor/snooper.py --checkpoint output/badnet-FF-10-0.97.hdf5`
    ```

    The final mask and pattern will be written to `mask.py` and `pattern.py`

## Results
TODO! Currently running TABOR for 500 epochs as in the paper takes about 4 hours on my hardware so generating the full results table is still upcoming.

## TODOs:
- [x] Implement BadNets
- [x] Implement image generation and basic categorical crossentropy optimization function
- [x] Implement all regularization terms from the TABOR paper  
- [ ] Add the outlier detection metric from section 4.2.4 from the TABOR paper
- [ ] Generate a results table