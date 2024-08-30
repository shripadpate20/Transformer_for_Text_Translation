# English to Marathi Transformer Translation Model

This project implements a Transformer-based neural machine translation model for translating English text to Marathi. It uses PyTorch and the Hugging Face datasets library to train and evaluate the model.

## Features

- Transformer architecture for sequence-to-sequence translation
- Custom dataset handling for bilingual data
- Tokenizer building and management
- Training with learning rate scheduling and early stopping
- Validation with various metrics (Character Error Rate, Word Error Rate, BLEU Score)
- TensorBoard integration for monitoring training progress
- Greedy decoding for inference

## Requirements

- Python 3.7+
- PyTorch
- torchtext
- datasets (Hugging Face)
- tokenizers
- torchmetrics
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/en-mr-transformer-translation.git
   cd en-mr-transformer-translation
   ```

2. Install the required packages:
   ```
   pip install torch torchtext datasets tokenizers torchmetrics tqdm
   ```

## Usage

1. Configure the model and training parameters in the `get_config()` function.

2. Run the training script:
   ```
   python transformers.py
   ```

3. Monitor training progress using TensorBoard:
   ```
   tensorboard --logdir=runs
   ```

## Model Architecture

The model uses a standard Transformer architecture with:
- Multi-head attention mechanisms
- Positional encoding
- Feed-forward neural networks
- Layer normalization

Key hyperparameters:
- d_model: 512
- Number of heads: 8
- Number of encoder/decoder layers: 6
- Dropout: 0.1

## Data

The model is trained on the OPUS Books dataset for English to Marathi translation. The dataset is automatically downloaded and processed using the Hugging Face datasets library.

## Training

The training process includes:
- Dataset splitting into training and validation sets (90% train, 10% validation)
- Custom data loading and batching
- Adam optimizer with custom learning rate
- Model checkpointing every 10 epochs

## Evaluation

The model is evaluated using:
- Character Error Rate (CER)
- Word Error Rate (WER)
- BLEU Score

Evaluation is performed on the validation set during training.

## Inference

The `greedy_decode` function can be used for inference on new English sentences. To use it:

1. Load a trained model
2. Prepare your input sentence
3. Tokenize the input
4. Call `greedy_decode` with your model and input

Example usage will be provided in future updates.

## Contributing

Contributions to improve the model or extend its functionality are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see below for details:

MIT License

Copyright (c) 2023 [shripad]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- This project uses the Transformer architecture as described in the paper "Attention Is All You Need" by Vaswani et al.
- The implementation is inspired by various open-source NMT projects and tutorials.
- Thanks to the Hugging Face team for providing easy-to-use datasets and tokenizers.
