# Description of Datasets and Model Implementations of all downtasks

## =====================================================
# 1. COPA Dataset and Model Implementation

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

# 2. NER Dataset and Model Implementation

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

# 3. Marathi News Category Classification

## Introduction
The task is to predict the genre or topic of a given news article or news headline. The news article category datasets are created using IndicCorp for 9 languages. The categories are determined from URL components and include generic categories that are likely to be consistent across websites (e.g., entertainment, sports, business, lifestyle, technology, politics, crime).

## Dataset
The Marathi news category dataset includes:
- **texts**: The news articles or headlines.
- **labels**: The categories corresponding to each news article or headline.

## Model Architecture

### MarathiDatasetCreate Class
The `MarathiDatasetCreate` class handles the dataset for the news category classification task. It uses a tokenizer and a FastText model to generate embeddings for the text data. The class initializes with lists of texts and labels, a FastText model for word embeddings, a dictionary mapping tokens to indices, and a language parameter. It also encodes the labels using `LabelEncoder`.

### Collate Function
The `collate_fn` function manages the batching of data for training or inference. It ensures that all sequences in the batch are padded to the same length, allowing for efficient batch processing by the model.

### BiLSTMClassifier Class
The `BiLSTMClassifier` class defines a neural network model for the classification task. The architecture includes:
- An LSTM layer to process the text sequences and capture their sequential information.
- The LSTM layer is bidirectional to capture both forward and backward dependencies in the text.
- A fully connected (FC) layer to map the output of the LSTM layers to the number of categories.
- The output of the LSTM layer at the last time step is passed through the FC layer to produce the final category scores.

## =====================================================

# 4. Wikipedia Section-title Prediction

## Introduction
The task is to predict the correct title for a Wikipedia section from a given list of four candidate titles (3 incorrect, 1 correct). The dataset is created using the open-source tool WikiExtractor to extract sections and their titles from Wikipedia. To make the task challenging, the 3 incorrect candidates for a given section are chosen only from the titles of other sections in the same article as the given section.

## Dataset
The Wikipedia section-title prediction dataset includes:
- **sectionText**: The text of a Wikipedia section.
- **titleA**: The first candidate title.
- **titleB**: The second candidate title.
- **titleC**: The third candidate title.
- **titleD**: The fourth candidate title.
- **correctTitle**: The key indicating the correct title among the four candidates (e.g., 'titleA').

## Model Architecture

### TitleSelectionDataset Class
The `TitleSelectionDataset` class handles the dataset for the section-title prediction task. It uses a tokenizer and a FastText model to generate embeddings for the text data. The class initializes with a DataFrame containing the dataset, a FastText model for word embeddings, a tokenizer for tokenizing the text, and a language parameter.

### Collate Function
The `collate_fn` function manages the batching of data for training or inference. It ensures that all sequences in the batch are padded to the same length, allowing for efficient batch processing by the model.

### TitleClassifier Class
The `TitleClassifier` class defines a neural network model for the classification task. The architecture includes:
- An ELMo model to generate contextualized embeddings for the section text and candidate titles.
- A classifier network consisting of:
  - A fully connected (FC) layer to reduce the dimensionality of the concatenated embeddings.
  - A ReLU activation function to introduce non-linearity.
  - A final FC layer to produce a score for each candidate title.
- The section embeddings and each candidate title's embeddings are averaged across the sequence length.
- The combined embeddings of the section text and each title are passed through the classifier to produce the final scores for the titles.




