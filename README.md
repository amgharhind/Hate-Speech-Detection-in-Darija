# Hate Speech Detection in Moroccan Darija using MARBERT

## Overview

This project addresses the critical challenge of detecting hate speech in the Moroccan Arabic dialect (Darija) by fine-tuning the **MARBERT** model. The goal is to contribute to safer online environments by accurately identifying toxic language specific to Darija. This README provides a comprehensive overview of the project, from the initial motivation and dataset to the model's performance and limitations.

## The "Why": Motivation and Importance

The proliferation of social media has led to a significant increase in online hate speech. While many models exist for detecting hate speech in major languages, low-resource dialects like Darija are often overlooked. Darija, with its unique blend of Arabic, Berber, French, and Spanish, presents a significant challenge for generic language models. This project aims to bridge that gap by developing a model specifically tailored to the nuances of Darija, thereby making online platforms safer and more inclusive for Moroccan users.

## Dataset: OMCD (Offensive Moroccan Comments Dataset)

The dataset used in this project is the **OMCD (Offensive Moroccan Comments Dataset)**, a collection of comments in Darija, labeled as either hate speech (1) or not hate speech (0). This dataset was specifically curated for Darija hate speech detection and was sourced from a Kaggle dataset [1].

**Dataset Statistics:**

| Dataset Split | Number of Comments |
| :--- | :--- |
| Training Set | 6,419 |
| Testing Set | 1,605 |
| **Total** | **8,024** |

**Class Distribution:**

| Class | Number of Comments |
| :--- | :--- |
| Hate Speech (1) | 4,304 |
| Not Hate Speech (0) | 3,720 |

This distribution indicates a slight imbalance, with a higher number of hate speech examples. This imbalance is handled during model training to prevent bias towards the majority class.

## Project Pipeline

The project follows a systematic pipeline to ensure the development of a robust and accurate hate speech detection model:

1.  **Data Loading and Exploration**: The training and testing datasets (`train.csv` and `test.csv`) are loaded and combined for a comprehensive analysis. Initial exploration includes examining the class distribution and the presence of emojis within the comments.
2.  **Data Preprocessing**: The text data undergoes a series of crucial preprocessing steps to clean and normalize it for the MARBERT model. This ensures that the model receives consistent and relevant input, improving its ability to learn patterns. The preprocessing steps, as detailed in the project's presentation, include:
    *   **Punctuation Removal**: All punctuation marks are removed from the comments to reduce noise and focus on the textual content. 
        *Example (from PPT)*: 
        - Before: `لخوت رجع معاها لواجب الوطني#`
        - After: `لخوت رجع معاها لواجب الوطني`
    *   **Emoji Handling**: Emojis are extracted and then removed from the text. While emojis can convey sentiment, for this project, the focus is on the textual content for hate speech detection.
        *Example (from PPT)*: 
        - Before: `لي جا من عند بيمو اخبط لأباك 🦁🦁🦁`
        - After: `لي جا من عند بيمو اخبط لأباك`
    *   **Single Digit Removal**: Numerical digits are removed from the text, as they typically do not contribute to the sentiment or hate speech classification.
        *Example (from PPT)*: 
        - Before: `لأن فرنسا كونت 14 امرأة على العلمانية`
        - After: `لأن فرنسا كونت امرأة على العلمانية`
    *   **Latin Character Handling (Transcription/Translation)**: Words written in Latin script are detected and replaced by their transcription or translation into Arabic. This step is crucial for models like MARBERT, which are optimized for Arabic script, ensuring that the model focuses solely on Darija text and avoids misinterpretations from transliterated words.
        *Example (from PPT)*: 
        - Before: `ضربة مرضكة و مبغاوش Lherba`
        - After: `لherba ضربة مرضكة و مبغاوش`
    *   **Extra Space Suppression**: Multiple spaces are replaced with a single space to standardize the text formatting.
        *Example (from PPT)*: 
        - Before: `هذا  النص  يحتوي على  مسافات  غير ضرورية`
        - After: `هذا النص يحتوي على مسافات غير ضرورية`
    *   **Text Normalization (Character Repetition)**: This involves handling elongated words by reducing repeated characters to a maximum of three, preventing issues with exaggerated spellings.
        *Example (from PPT - Case 1)*: 
        - Before: `ياااااا سلام`
        - After: `يااا سلام`
        *Example (from PPT - Case 2)*: 
        - Before: `هههههههههه`
        - After: `ضحك`
3.  **Data Splitting Strategy**: The dataset is initially provided with a pre-defined training and testing split (`train.csv` and `test.csv`). This split is maintained to ensure consistency with the original dataset's evaluation methodology. The notebook `hate-speech-detection-marbert-without-dense-lay.ipynb` further splits the training data into training (65%), validation (15%), and test (20%) sets using a stratified approach to preserve the distribution of labels.
    ```python
    # Step 1: Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Step 2: Stratified split into training (65%) and temp (35%)
    train_data, temp_data = train_test_split(
        df, test_size=0.35, stratify=df['off'], random_state=42
    )
    # Step 3: Stratified split of temp (35%) into validation (15%) and test (20%)
    val_data, test_data = train_test_split(
        temp_data, test_size=20/(15+20), stratify=temp_data['off'], random_state=42
    )
    ```
4.  **Model Selection and Fine-Tuning**: The pre-trained **MARBERTv2** model is chosen as the base model due to its strong performance on Arabic NLP tasks. The model is then fine-tuned on the preprocessed Darija hate speech dataset. The fine-tuning process adapts the general language understanding capabilities of MARBERT to the specific task of identifying hate speech in Darija. The model used is `AutoModelForSequenceClassification` from the `transformers` library, initialized with `UBC-NLP/MARBERT` and configured for binary classification (`num_labels=2`).
5.  **Model Training and Evaluation**: The fine-tuned model is trained on the training set and evaluated on the test set using various metrics such as accuracy, precision, recall, and F1-score.

## Project Structure and Files

```
Hate-Speech-Detection-in-Darija/
├── hate-speech-detection-marbert_preprocessing.ipynb  # Notebook for data preprocessing and cleaning
├── hate-speech-detection-marbert-without-dense-lay.ipynb # Main notebook for model training and evaluation
├── hate_speech_detection_darija.pptx                  # Presentation slides detailing the project
└── README.md                                          # This file
```

## Results and Explanations

*(Note: The following results are based on the model's performance as detailed in the `hate-speech-detection-marbert-without-dense-lay.ipynb` notebook and the `hate_speech_detection_darija.pptx` presentation.)*

The fine-tuned MARBERT model achieved the following performance metrics on the test set:

| Metric | Score |
| :--- | :--- |
| Accuracy | 0.89 |
| Precision | 0.91 |
| Recall | 0.87 |
| F1-Score | 0.89 |

**Explanation of Results**:

*   **Accuracy (0.89)**: The model correctly classified 89% of the comments. This indicates a strong overall performance in distinguishing between hate speech and non-hate speech.
*   **Precision (0.91)**: When the model predicted a comment as hate speech, it was correct 91% of the time. This is a crucial metric for hate speech detection, as it minimizes false positives, ensuring that legitimate speech is not incorrectly flagged.
*   **Recall (0.87)**: The model successfully identified 87% of all actual hate speech comments in the dataset. A high recall is important to ensure that most instances of hate speech are caught.
*   **F1-Score (0.89)**: The F1-score, which is the harmonic mean of precision and recall, indicates a good balance between the two metrics, suggesting a robust model that performs well across both identifying hate speech and avoiding false alarms.

## Limitations and Future Work

As highlighted in the project presentation (`hate_speech_detection_darija.pptx`), several limitations were identified:

*   **Dataset Size**: While the dataset is of a reasonable size, a larger and more diverse dataset could further improve the model's performance and generalization capabilities. A broader range of hate speech expressions and non-hate speech contexts would enhance the model's robustness.
*   **Sarcasm and Irony**: The model may struggle with nuanced forms of hate speech, such as sarcasm and irony, which are challenging to detect even for humans. These linguistic complexities often require a deeper understanding of context and intent that current models may not fully capture.
    *   **Example of Sarcasm (from PPT)**: A comment like `واو، شحال زوين هاد الكلام!` (Wow, how beautiful this speech is!) could be sarcastic hate speech if the context implies the opposite.
*   **Evolving Language**: Darija, like any living language, is constantly evolving. New slang, expressions, and ways of communicating hate speech emerge over time. The model may need to be periodically retrained to keep up with these linguistic shifts and maintain its effectiveness.
    *   **Example of Evolving Slang (from PPT)**: A new derogatory term might emerge that the model, trained on older data, would not recognize as hate speech.

**Future work** could involve:

*   Expanding the dataset to include more diverse examples of hate speech, particularly those involving sarcasm and irony.
*   Exploring more advanced model architectures and fine-tuning techniques, potentially incorporating multi-modal information or more sophisticated contextual embeddings.
*   Developing a real-time hate speech detection system that can be integrated into social media platforms, with mechanisms for continuous learning and adaptation to evolving language.

## Examples of Hate Speech Detection

Here are some examples of how the model might classify different types of comments, drawing from the provided dataset and presentation:

| Comment (Darija) | Translation | Predicted Label |
| :--- | :--- | :--- |
| `هادشي غير تخربيق` | `This is just nonsense` | Hate Speech (1) |
| `الله يهديك` | `May God guide you` | Not Hate Speech (0) |
| `كلامك زوين بزاف` | `Your words are very beautiful` | Not Hate Speech (0) |
| `واش هادشي معقول` | `Is this reasonable?` | Not Hate Speech (0) |

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/amgharhind/Hate-Speech-Detection-in-Darija.git
    cd Hate-Speech-Detection-in-Darija
    ```
2.  **Install dependencies:**
    ```bash
    pip install transformers torch scikit-learn pandas jupyter emoji
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Navigate to the `.ipynb` files to explore the code and run the models.

## License

This project is open-source and available under the [MIT License](LICENSE).

## References

[1] OMCD: Offensive Moroccan Comments Dataset [https://link.springer.com/article/10.1007/s10579-023-09663-2](https://link.springer.com/article/10.1007/s10579-023-09663-2)


