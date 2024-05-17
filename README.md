# Spotify Song Mood Classification Project

Welcome to the Spotify Song Mood Classification project! This repository contains a series of Jupyter notebooks designed to classify Spotify songs by mood, using machine learning models. The project aims to create a large, labeled dataset of songs categorized by mood, which can then be utilized for various applications such as playlist generation.

## Overview

The project is structured into three main phases, each corresponding to a notebook:

1. **Concatenation of Two Labeled Mood Song Datasets (`concat_datasets.ipynb`):** This notebook combines two distinct datasets into a unified, labeled dataset suitable for training classification models.

2. **Mood Classification of Spotify Songs with Machine Learning (`main.ipynb`):** Utilizes the concatenated dataset to train and evaluate machine learning models, ultimately labeling a large, unlabeled dataset of Spotify songs.

3. **Refining the Main Labeled Dataset for Primary Artist Focus (`clean_labeled_dataset.ipynb`):** Cleans the labeled dataset to focus solely on the primary artist, enhancing usability for playlist generation and other applications. It also provides a quick subjective evaluation of the labeled dataset.

## Getting Started

To get started with this project, please follow the steps below:

### Step 1: Install the Requried Libraries

Please install the required libraries via the `requirements.txt` file.

### Step 2: Download the main Dataset

Please download the `main_dataset.csv` file from Kaggle's [6K Spotify Playlist](https://www.kaggle.com/datasets/viktoriiashkurenko/278k-spotify-songs?select=main_dataset.csv) dataset.

### Step 3: Concatenate Datasets

Run the `concat_datasets.ipynb` notebook first. This notebook will merge two separate datasets into one, creating the training data for the classification model. Ensure that both source datasets are available in the specified paths within the notebook.

### Step 4: Train the Classification Model

Next, proceed with the `main.ipynb` notebook. This notebook trains a machine learning model on the concatenated dataset and labels a larger, unlabeled dataset with mood categories. The default classifier is Random Forest, but instructions are provided if you wish to switch to XGBoost.

### Step 5: Clean the Labeled Dataset

Finally, use the `clean_labeled_dataset.ipynb` notebook to refine the labeled dataset, focusing on data related to the main artist. This step simplifies the dataset, making it more manageable for downstream tasks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Juan Antonio Robledo** - _Initial Work_ - [JuanRobledo12](https://github.com/JuanRobledo12)

## Acknowledgments

A special thanks to the creators of the original datasets used in this project:

- [AISongRecommender](https://github.com/michaelmoschitto/AISongRecommender/tree/main)
- [Spotify Machine Learning Project](https://github.com/cristobalvch/Spotify-Machine-Learning)
- [6K Spotify Playlist](https://www.kaggle.com/datasets/viktoriiashkurenko/278k-spotify-songs?select=main_dataset.csv)

We appreciate their efforts in making these resources available to the community.
