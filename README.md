# PepSpace Visualization

PepSpace is an interactive Streamlit application for exploring bioactive peptide embedding spaces. It visualizes peptide datasets with UMAP and densMAP, supports default Veltri data paths, and allows users to upload custom embedding and metadata files.

## Features

* Interactive 2D UMAP and densMAP projections
* AMP vs non-AMP coloring
* Hover inspection for peptide IDs, sequences, and labels
* Upload custom embeddings (`.npy`) and metadata (`.csv`)
* Metadata source switching between processed Veltri metadata and raw Veltri CSV
* Optional 3D sphere projection
* Adjustable UMAP and densMAP parameters from the sidebar
* Cached data loading and projection runs for faster interaction

The repository also includes an advanced prototype, `app_test.py`, with additional controls for n-dimensional sphere projection, 2D region selection, trustworthiness metrics, and saving selected peptides.

## Python Version

Python 3.12 is recommended. This project was developed with Python 3.12.10.

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

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the main Streamlit app:

```bash
streamlit run app.py
```

To run the advanced prototype:

```bash
streamlit run app_test.py
```

After launching, Streamlit will print a local URL and usually open the application in your browser automatically.

## Data

The main app expects precomputed Veltri embeddings and metadata at these paths:

```text
embeddings/veltri/veltri_embeddings.npy
embeddings/veltri/veltri_metadata.csv
```

It can also read the raw Veltri CSV from:

```text
data/veltri/all_veltri.csv
```

If the default embedding files are not present, restore them before using the default dataset, generate them with the scripts in `scripts/`, or upload custom files through the sidebar.

Custom embeddings must be NumPy arrays:

```text
.npy
shape: (samples, embedding_dimension)
```

Custom metadata must be a CSV with the same number of rows as the embedding array. The app auto-detects common column names for peptide sequence, ID, label, and AMP status.

Common metadata columns:

```text
id, sequence, label, is_amp
```

or:

```text
peptide_id, sequence, source, AMP
```

## Example Workflow

1. Load the default Veltri embeddings, or upload custom embeddings and metadata.
2. Choose Veltri metadata or the raw Veltri CSV for labels and hover text.
3. Adjust UMAP or densMAP parameters in the sidebar.
4. Explore the 2D projection and inspect peptides with hover text.
5. Optionally enable the 3D sphere projection.

## Project Structure

```text
Pepspace-Visualization/
  app.py                         Main Streamlit visualization app
  app_test.py                    Advanced/experimental Streamlit app
  requirements.txt               Python dependencies
  README.md                      Project documentation
  LICENSE                        Project license

  data/veltri/                   Raw Veltri dataset
  embeddings/                    Precomputed embeddings and metadata
  scripts/                       Data preparation, embedding, and optimization scripts
  utils/                         Shared utility modules
  results/                       UMAP optimization result files
```

## Scripts

Useful scripts in this repository:

```text
scripts/make_veltri_embeddings.py    Generate Veltri embedding files
scripts/optimize_umap.py             Run UMAP parameter optimization
scripts/veltri_dataset.py            Prepare Veltri dataset inputs
scripts/make_fake_csv.py             Create sample fake metadata
scripts/make_real_csv.py             Create sample real metadata
```

## Methods

UMAP is used for nonlinear dimensionality reduction of peptide embedding vectors.

densMAP is a density-preserving extension of UMAP that can better maintain relative cluster density.

Protein language model embeddings convert peptide sequences into numerical vectors suitable for projection and visualization.

The optional sphere projection normalizes higher-dimensional UMAP outputs and projects them into 3D for interactive Plotly visualization.

## Usage With Custom Datasets

Upload your own peptide dataset from the sidebar:

1. Upload embeddings as a `.npy` file.
2. Upload metadata as a `.csv` file.
3. Select sequence, ID, or label columns manually if auto-detection fails.
4. Run UMAP or densMAP visualization.

The metadata row count should match the number of embedding rows. If the counts differ, the app warns you and truncates to the shorter length.

## Author

Alejandro Lopez and Brian Andrade<br>
Department of Computer Science<br>
California State University - Los Angeles
