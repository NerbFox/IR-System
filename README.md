<h2 align="center">
Information Retrieval System with BERT Embeddings<br/>
</h2>

## Table of Contents
1. [General Info](#general-information)
2. [Creator Info](#creator-information)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Setup](#setup)


<a name="general-information"></a>

## General Information
A simple Python-based GUI built with Qt (using PySide2/PySide6 or PyQt5/PyQt6) and styled with qt_material. This application functions as an information-retrieval system—allowing users to enter or batch-process queries, choose TF/IDF weighting options, expand queries, and retrieve relevant documents via an inverted-file index.

<a name="creator-information"></a>

## Creator Information

| Nama                        | NIM      |
| --------------------------- | -------- | 
| Nigel Sahl                  | 13521043 |
| Alex Sander                 | 13521061 |
| Bintang Dwi Marthen         | 13521144 |
| Hanif Muhammad Zhafran      | 13521157 |
| Mohammad Rifqi Farhansyah   | 13521166 |

<a name="features"></a>

## Features

- Read `input` directly (single-query mode) or load queries from a file (batch mode)
- Select a directory as the `source of documents`
- Enable `Mods` options: Stemming and Eliminate Stop Words
- Choose one `weighting` scheme: TF, IDF, TF×IDF, or TF×IDF×Normalized
- Select a `TF method`: Logarithmic, Raw, Binary, or Augmented
- Specify a fixed number of `terms` to add to the query
- Display the `expanded query` in the GUI
- View the `inverted‐file` index in a separate window
- `Switch themes` on the fly via the “Styles” menu
- Pick a `visualization color` (as defined in the theme’s extra palette)

<a name="technologies-used"></a>

## Technologies Used
- PySide6 (Qt for Python GUI)
- qt-material (Qt Material Design themes)
- NumPy (numerical computing)
- CuPy (GPU-accelerated array computing)
- scikit-learn (machine learning)
- SciPy (scientific computing)
- NLTK (natural language processing)
- transformers (Hugging Face BERT and other models)
- torch (PyTorch, deep learning)
- tokenizers (Hugging Face tokenization)
- Jinja2 (templating)
- requests (HTTP requests)
- tqdm (progress bars)
- sympy (symbolic mathematics)
- networkx (graph analysis)
- pytest (testing)
- PyYAML (YAML parsing)
- Other utilities: packaging, filelock, fsspec, joblib, safetensors, typing_extensions, charset-normalizer, idna, urllib3, certifi, click, colorama, iniconfig, MarkupSafe, mpmath, pluggy, setuptools, shiboken6, threadpoolctl

<a name="setup"></a>

## Setup
1. Clone this repository
2. Download all the required packages by running `pip install -r requirements.txt` in the terminal.
3. Go to `gui` directory using the command `cd gui`
4. Run the GUI using the command below `python main.py`