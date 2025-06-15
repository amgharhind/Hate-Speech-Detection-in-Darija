# Hate Speech Detection in Moroccan Darija using MARBERT

This project focuses on detecting hate speech in the Moroccan Arabic dialect (Darija) by fine-tuning  **MARBERT** model. 

## Overview

Detecting toxic language online is crucial for creating safer digital spaces. This project addresses this challenge by fine-tuning the powerful MARBERT model, pre-trained on a large corpus of Arabic text, to understand the specific nuances of hate speech in Darija. 

## Methodology

The project follows a structured methodology to optimize the model for the task:

1.  **Darija-Specific Preprocessing:** A dedicated notebook (`hate-speech-detection-marbert-v2_preprocessing.ipynb`) handles the cleaning and preparation of Darija text data to make it suitable for the MARBERT model.
2.  **MARBERT Fine-Tuning:** The project fine-tunes the `MARBERT` model on the preprocessed hate speech dataset.


## Technology 

-   **Model:** MARBERTv2
-   **Core Libraries:** Hugging Face `transformers`, PyTorch, Scikit-learn, Pandas

