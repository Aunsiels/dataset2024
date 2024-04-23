# LM-KBC: Knowledge Base Construction from Pre-trained Language Models (3rd Edition)

This repository hosts data for the LM-KBC challenge at ISWC
2024 (https://lm-kbc.github.io/challenge2024/).

This repository contains:

- The dataset for the challenge
- Evaluation script
- Baselines
- Instructions for submitting your predictions

## Table of contents

1. [News](#news)
2. [Challenge overview](#challenge-overview)
3. [Dataset](#dataset)
4. [Evaluation metrics](#evaluation-metrics)
5. [Getting started](#getting-started)
    - [Setup](#setup)
    - [Baselines](#baselines)
    - [How to structure your prediction file](#how-to-structure-your-prediction-file)

## News

- 22.4.2024: Release of dataset v1.0
- 25.3.2024: Release of preliminary evaluation script and GPT-baseline

## Challenge overview

Pretrained language models (LMs) like ChatGPT have advanced a range of semantic
tasks and have also shown promise for
knowledge extraction from the models itself. Although several works have
explored this ability in a setting called
probing or prompting, the viability of knowledge base construction from LMs
remains under-explored. In the 3rd edition
of this challenge, we invite participants to build actual disambiguated
knowledge bases from LMs, for given subjects and
relations. In crucial difference to existing probing benchmarks like
LAMA ([Petroni et al., 2019](https://arxiv.org/pdf/1909.01066.pdf)), we make no
simplifying assumptions on relation
cardinalities, i.e., a subject-entity can stand in relation with zero, one, or
many object-entities. Furthermore,
submissions need to go beyond just ranking predicted surface strings and
materialize disambiguated entities in the
output, which will be evaluated using established KB metrics of precision and
recall.

> Formally, given the input subject-entity (s) and relation (r), the task is to
> predict all the correct
> object-entities ({o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>k</sub>}) using LM
> probing.

## Dataset

Number of unique subject-entities in the data splits.

<table>
<thead>
    <tr>
        <th>Relation</th>
        <th>Train</th>
        <th>Val</th>
        <th>Test</th>
        <th>Special features</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>countryLandBordersCountry</td>
        <td>63</td>
        <td>63</td>
        <td>63</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>personHasCityOfDeath</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
    <tr>
        <td>seriesHasNumberOfEpisodes</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Object is numeric</td>
    </tr>
    <tr>
        <td>awardWonBy</td>
        <td>10</td>
        <td>10</td>
        <td>10</td>
        <td>Many objects per subject</td>
    </tr>
    <tr>
        <td>companyTradesAtStockExchange</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>Null values possible</td>
    </tr>
</tbody>
</table>

## Evaluation metrics

We evaluate the predictions using macro precision, recall, and F1-score.
See the evaluation script ([evaluate.py](evaluate.py)) for more details.

```bash
python evaluate.py \
  -g data/val.jsonl \
  -p data/testrun-XYZ.jsonl
```

Parameters: ``-g`` (the ground truth file), ``-p`` (the prediction file).

## Getting started

### Setup

1. Clone this repository:

    ```bash
    mkdir lm-kbc-2024
    cd lm-kbc-2024
    git clone https://github.com/lm-kbc/dataset2024.git
    cd dataset2024
    ```

2. Create a virtual environment and install the requirements:

    ```bash
    conda create -n lm-kbc-2024 python=3.12.1
    ```

    ```bash
    conda activate lm-kbc-2024
    pip install -r requirements.txt
    ```

3. Write your own solution and generate predictions (format described
   in [How to structure your prediction file](#how-to-structure-your-prediction-file)).
4. Evaluate your predictions using the evaluation script (
   see [Evaluation metrics](#evaluation-metrics)).
5. Submit your solutions to the organizers (
   see [Call for Participants](https://lm-kbc.github.io/challenge2024/#call-for-participants)).

### Baselines

We provide baselines using Masked Language
Models ([models/baseline_fill_mask_model.py](models/baseline_fill_mask_model.py))
and Autoregressive Language
Models ([models/baseline_generation_model.py](models/baseline_generation_model.py)).

You can run these baselines via the [baseline.py](baseline.py) script and
providing it with the corresponding configuration file. We provide example
configuration files for the baselines in the [configs](configs) directory.

- `bert-large-cased` (Masked Language Model)
    ```bash
    python baseline.py -c configs/baseline-bert-large-cased.yaml -i data/val.jsonl
    python evaluate.py -g data/val.jsonl -p output/baseline-bert-large-cased.jsonl
    ```
  Results:
    ```text
                                      p      r     f1
    awardWonBy                    0.300  0.000  0.000
    companyTradesAtStockExchange  0.000  0.350  0.000
    countryLandBordersCountry     0.632  0.702  0.487
    personHasCityOfDeath          0.290  0.630  0.242
    seriesHasNumberOfEpisodes     1.000  0.000  0.000
    *** Average ***               0.444  0.336  0.146
    ```

- `facebook/opt-1.3b` (Autoregressive Language Model, quantized)
    ```bash
    python baseline.py -c configs/baseline-opt-1.3b.yaml -i data/val.jsonl
    python evaluate.py -g data/val.jsonl -p output/baseline-opt-1.3b.jsonl
    ```
  Results:
    ```text
                                      p      r     f1
    awardWonBy                    0.100  0.006  0.011
    companyTradesAtStockExchange  0.260  0.441  0.242
    countryLandBordersCountry     0.230  0.395  0.216
    personHasCityOfDeath          0.270  0.490  0.270
    seriesHasNumberOfEpisodes     0.000  0.000  0.000
    *** Average ***               0.172  0.266  0.148
    ```

- `meta-llama/llama-2-7b-hf` (Autoregressive Language Model, quantized)
    ```bash
    export HUGGING_FACE_HUB_TOKEN=your_token
    python baseline.py -c configs/baseline-llama-2-7b-hf.yaml -i data/val.jsonl
    python evaluate.py -g data/val.jsonl -p output/baseline-llama-2-7b-hf.jsonl
    ```
  Results:
    ```text
                                      p      r     f1
    awardWonBy                    0.362  0.011  0.021
    companyTradesAtStockExchange  0.340  0.528  0.314
    countryLandBordersCountry     0.638  0.731  0.568
    personHasCityOfDeath          0.320  0.590  0.320
    seriesHasNumberOfEpisodes     0.000  0.000  0.000
    *** Average ***               0.332  0.372  0.245
    ```

- `meta-llama/Meta-Llama-3-8B` (Autoregressive Language Model, quantized)
    ```bash
    export HUGGING_FACE_HUB_TOKEN=your_token
    python baseline.py -c configs/baseline-llama-3-8b.yaml -i data/val.jsonl
    python evaluate.py -g data/val.jsonl -p output/baseline-llama-3-8b.jsonl
    ```
  Results:
    ```text
                                      p      r     f1
    awardWonBy                    0.000  0.000  0.000
    companyTradesAtStockExchange  0.540  0.688  0.518
    countryLandBordersCountry     0.625  0.770  0.605
    personHasCityOfDeath          0.430  0.650  0.430
    seriesHasNumberOfEpisodes     0.000  0.000  0.000
    *** Average ***               0.319  0.422  0.311
    ```

### How to structure your prediction file

Your prediction file should be in the jsonl format.
Each line of a valid prediction file contains a JSON object which must
contain at least 3 fields to be used by the evaluation script:

- ``SubjectEntity``: the subject entity (string)
- ``Relation``: the relation (string)
- ``ObjectEntitiesID``: the predicted object entities ID, which should be a list
  of Wikidata IDs (strings).

This is an example of how to write a prediction file:

```python
import json

# Dummy predictions
predictions = [
    {
        "SubjectEntity": "Dominican republic",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q790", "Q717", "Q30", "Q183"]
    },
    {
        "SubjectEntity": "Eritrea",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": ["Q115"]
    },
    {
        "SubjectEntity": "Estonia",
        "Relation": "CountryBordersWithCountry",
        "ObjectEntitiesID": []
    }

]

fp = "./path/to/your/prediction/file.jsonl"

with open(fp, "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")
```
