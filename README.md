<h2 align="center">
Information Retrieval System with BERT Embeddings<br/>
</h2>

## Table of Contents
1. [General Info](#general-information)
2. [Creator Info](#creator-information)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Setup](#setup)
<!-- 8. [Screenshots](#screenshots) -->
6. [Structure](#structure)


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
- PyQt
- Qt-Material
- NumPy and CuPy

<a name="setup"></a>

## Setup
1. Download all the requirements
2. Clone this repository to your local directory by using this command in your terminal
```bash
git clone https://github.com/rifqifarhansyah/Tubes2_dicarryVieridanZaki.git
```
3. Go to `gui` directory using the command below
```bash
cd .\gui\
```
4. Run the model using a command below
```bash
python main.py --pyside6
```

<!-- <a name="screenshots"></a>

## Screenshots
<p>
  <img src="/img/SS1.png/">
  <p>Figure 1. Config File (*txt)</p>
  <nl>
  <img src="/img/SS2.png/">
  <p>Figure 2. Initial Appearance of the Program</p>
  <nl>
  <img src="/img/SS3.png/">
  <p>Figure 3. Result</p>
  <nl>
</p> -->

<a name="structure"></a>

## Structure
```bash
│   .gitignore
│   conftest.py
│   README.md
│   requirements.txt
│   __init__.py
│
├───gui
│   │   inverted_window.ui
│   │   main.py
│   │   main_window.ui
│   │   my_theme.xml
│   │   queryexp_window.ui
│   │   README.md
│   │   result_window.ui
│   │
│   └───img
│           logo.png
│
├───tests
│       test_bert_calculation.py
│       test_map_calculation.py
│       test_map_tf_idf_calculation.py
│       test_tf_idf_calculation.py
│       __init__.py
│
└───utils
        bert_calculation.py
        cuda_utils.py
        map_calculation.py
        tf_idf_calculation.py
        __init__.py
```
