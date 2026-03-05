# PepSpace Visualization

PepSpace is an interactive data visualization tool for exploring **bioactive peptide embedding spaces**.
It allows researchers to visualize peptide datasets using **UMAP** and **densMAP** dimensionality reduction techniques.

The application is built with **Python and Streamlit** and supports both built-in datasets and user-uploaded embeddings.

---

## Features

* Interactive visualization of peptide embedding spaces
* **UMAP** and **densMAP** projection methods
* AMP vs Non-AMP color visualization
* Upload custom embeddings (`.npy`) and metadata (`.csv`)
* Hover inspection of peptide sequences and IDs
* Adjustable UMAP parameters in real time

---

## Python Version

This project was developed with:

Python 3.12.10

---

## Installation

Clone the repository:

```bash
git clone https://github.com/brinstussymethod/Pepspace-Visualization.git
cd Pepspace-Visualization
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

After launching, the visualization will automatically open in your browser.

---

## Data

This repository includes **precomputed peptide embeddings** so the visualization can run immediately without regenerating embeddings.

Included dataset:

```
embeddings/veltri/veltri_embeddings.npy
embeddings/veltri/veltri_metadata.csv
```

These embeddings were generated using **protein language models** for the Veltri antimicrobial peptide dataset.

Users can also upload their own datasets directly through the interface.

Required formats:

Embeddings:

```
.npy
shape: (samples, embedding_dimension)
```

Metadata:

```
.csv
id, sequence, label/is_amp
```

The number of rows in the metadata must match the number of embeddings.

---

## Example Visualization Workflow

1. Load default Veltri dataset
2. Run **UMAP** or **densMAP** projection
3. Explore clusters of peptides in embedding space
4. Inspect peptide sequences using hover information

---

## Project Structure

```
Pepspace-Visualization

app.py                    Streamlit visualization app
requirements.txt          Python dependencies
README.md                 Project documentation

data/                     Raw datasets
embeddings/               Precomputed embeddings
scripts/                  Data preparation and embedding scripts
notebooks/                Experiment notebooks
src/                      Utility modules
vis/                      Visualization helpers
```

---

## Methods

This tool uses the following methods for embedding visualization:

UMAP
Uniform Manifold Approximation and Projection for dimension reduction.

densMAP
A density-preserving extension of UMAP for more faithful cluster density representation.

Protein Language Model Embeddings
Peptide sequences are converted into numerical vectors using pretrained protein models.

---

## Usage With Custom Datasets

You can upload your own peptide datasets through the sidebar:

1. Upload embeddings (`.npy`)
2. Upload metadata (`.csv`)
3. Select columns if auto-detection fails
4. Run UMAP/densMAP visualization

---

## Author

Alejandro Lopez and Brian Andrade 
Department of Computer Science 
California State Univeristy - Los Angeles
