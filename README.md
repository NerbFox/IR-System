<h2 align="center">
Information Retrieval System with BERT Embeddings<br/>
</h2>

## Table of Contents
1. [General Info](#general-information)
2. [Creator Info](#creator-information)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Setup](#setup)
6. [User Manual](#user-manual)


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

## User Manual

Dari tampilan utama aplikasi GUI, pilih pengaturan yang ingin diimplementasikan pada document dan query
![image](https://github.com/user-attachments/assets/727bf876-d2df-4f4d-94f0-e1ba0736d03c)

Tekan tombol ‘SELECT FILE’ pada bagian ‘DOCUMENT SOURCE’ dan pilih file yang akan menjadi corpus source document
![image](https://github.com/user-attachments/assets/e1f2d351-1921-42fa-b81c-55c8ab5fa644)

Tekan tombol ‘SELECT RELEVANT DOCUMENT FILE’ pada bagian ‘INPUT QUERY’ dan pilih file yang akan menjadi informasi dokumen relevan
![image](https://github.com/user-attachments/assets/4bcd5025-0fe4-4105-bc64-e0822acaca6c)

Pilih jenis input yang ingin digunakan (‘SINGLE INPUT’ atau ‘BATCH INPUT’) pada bagian ‘INPUT QUERY’
Untuk jenis single input, ketikkan query yang akan diproses dan klik tombol ‘PROCESS’
![image](https://github.com/user-attachments/assets/506e49e2-5b10-4357-a71e-448398d6e87c)

Untuk jenis batch input, klik tombol ‘SELECT FILE’ pada bagian ‘INPUT QUERY’ dan pilih file yang mengandung query-query yang akan diproses
![image](https://github.com/user-attachments/assets/dafb95de-f62c-49ff-84b7-2b806764f2da)

3.X & 4.X. Jika ingin mengubah pengaturan yang ingin diimplementasikan, ulangi langkah dari langkah 1.
Untuk menampilkan inverted file dari document source yang telah dipilih, klik tombol ‘CHECK INVERTED FILE’ pada bagian bawah kanan layar GUI.
![image](https://github.com/user-attachments/assets/920bd831-8ec5-4c2b-b7e8-4abdc7d46157)

Untuk menampilkan inverted file suatu dokumen tertentu, pilih index dokumen yang ingin diperiksa inverted filenya dan klik tombol ‘PROCESS’
![image](https://github.com/user-attachments/assets/ee3fdfc1-219f-4f84-a707-97b33ae97aa0)

Untuk melanjutkan proses query expansion dan relevant document fetching, klik tombol ‘EXPAND QUERY’ pada bagian bawah kiri layar GUI
![image](https://github.com/user-attachments/assets/82ae4801-eed5-4413-9c68-643258bc6635)

GUI akan menampilkan layar baru yang mengandung query input yang telah di-expand.
![image](https://github.com/user-attachments/assets/92f28a5a-6bdf-4012-8446-1b71e95479cc)

Untuk melanjutkan proses relevant document fetching, klik tombol ‘RETRIEVE DOCUMENT’
![image](https://github.com/user-attachments/assets/5cb61542-6561-4b9c-b906-985769da64b6)

GUI akan menampilkan layar baru yang mengandung hasil fetching dokumen relevan untuk query asal dan query yang telah di-expand
![image](https://github.com/user-attachments/assets/5579b629-8f73-48be-bc72-06c7de2bf98d)

Untuk menampilkan hasil fetching dokumen relevan query tertentu, pilih index query yang ingin diperiksa (0 untuk single input) dan klik tombol ‘PROCESS’
![image](https://github.com/user-attachments/assets/6a3966ac-f510-4d04-a51a-46ef9c8ecf61)
