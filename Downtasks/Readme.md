# Description of Datasets and Model Implementations of all downtasks

## ===========================================================
# COPA Dataset and Model Implementation

## Introduction
The Choice Of Plausible Alternatives (COPA) task evaluates open-domain commonsense causal reasoning. This task consists of a large set of 2-choice questions, formulated as a premise and two alternatives written as sentences. The goal is to select the alternative that is more plausibly the cause (or effect) of the situation described by the premise. The dataset has been translated into three Indic languages: Hindi (hi), Marathi (mr), and Gujarati (gu).

## Dataset
The COPA dataset includes the following fields:
- **premise**: The given situation or statement.
- **choice1**: The first alternative that could be a cause or effect of the premise.
- **choice2**: The second alternative that could be a cause or effect of the premise.
- **label**: The correct alternative (0 for choice1, 1 for choice2).

## Model Architecture

### COPADataset Class
The `COPADataset` class handles the dataset for the COPA task. It uses a tokenizer and a FastText model to generate embeddings for the text data. The class initializes with a DataFrame containing the dataset, a FastText model for word embeddings, a tokenizer for tokenizing the text, and a language parameter to specify the language of the text data.

### Collate Function
The `collate_fn` function manages the batching of data for training or inference. It ensures that all sequences in the batch are padded to the same length, allowing for efficient batch processing by the model.

### ChoiceComparisonModel Class
The `ChoiceComparisonModel` class defines a neural network model for comparing the premise with the two choices. The architecture includes:
- An LSTM layer to process the text sequences and capture their sequential information.
- A fully connected (FC) layer to combine the output of the LSTM layers for the premise and the two choices.
- The LSTM layers are bidirectional and consist of two layers to capture both forward and backward dependencies in the text.
- The output of the LSTM layers is averaged across the sequence length to create fixed-size representations of the premise, choice1, and choice2.
- These fixed-size representations are concatenated and passed through the FC layer to produce the final logits for classification.

## =====================================================

# NER Dataset and Model Implementation

## Introduction
Named Entity Recognition (NER) involves identifying and classifying named entities in text into predefined categories such as Person (PER), Organisation (ORG), and Location (LOC). This implementation uses the WikiAnn NER dataset, which contains NER data for 282 languages. The dataset is created from Wikipedia by utilizing cross-language links to propagate English named entity labels to other languages.

## Dataset
The NER dataset includes the following fields:
- **tokens**: The list of tokens (words) in a sentence.
- **ner_tags**: The list of NER tags corresponding to each token.

## Model Architecture

### NERDataset Class
The `NERDataset` class handles the dataset for the NER task. It uses a tokenizer and a FastText model to generate embeddings for the text data. The class initializes with a DataFrame containing the dataset, a FastText model for word embeddings, a tokenizer for tokenizing the text, and a dictionary mapping tags to indices.

### Collate Function
The `collate_fn` function manages the batching of data for training or inference. It ensures that all sequences in the batch are padded to the same length, allowing for efficient batch processing by the model.

### NERModel Class
The `NERModel` class defines a neural network model for the NER task. The architecture includes:
- An LSTM layer to process the text sequences and capture their sequential information.
- A fully connected (FC) layer to map the output of the LSTM layers to the number of tags.
- The LSTM layer is bidirectional to capture both forward and backward dependencies in the text.
- The output of the LSTM layer is passed through the FC layer to produce the final tag scores for each token.

## =====================================================

