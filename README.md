# geoguessr-AI

A simple ViT-based streetview classifier.

## Usage

### Step 1. Install the requirements

```bash
python -m pip install -r requirements.txt
```

### Step 2. Configure environment variables

Put your Google Map API key in the file `.env`.

```bash
echo 'MAP_API="YOUR_GM_API_KEY"' >> ./.env
```

### Step 3. Fetch data

#### Step 3-1. Configure constants

Modify the file `const.py`

#### Step 3-2. Start fetching

```bash
python ./fetch_data.py
```

### Step 4. Train model

The arguments of `train.py` are as follows:

 - `-h`, `--help`: Show this help message and exit.
 - `-e`, `--num_epochs`: Number of epochs for training.
 - `-b`, `--batch_size`: Batch size for training.
 - `-lr`, `--learning_rate`: Learning rate for the optimizer.
 - `-s`, `--seed`: Random seed of train/test split.
 - `-dr`, `--dropout_rate`: Dropout rate for training.

Here's one example:

```bash
python ./train.py -b 100 -lr 1e-5 -s 114514 -e 10 -dr 0.3
```

### Step 5. Demo

```
python ./demo.py -m {model_weight_file}
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
