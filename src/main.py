import warnings

import colorama
import pandas as pd
import torch

from Utility import Utilities
from anomaly_detection import anomaly_detector
from features_engineering import features_engineering
from features_extracting import features_extracting
from logdata_read import logdata_read
from model_evaluation import model_evaluation

# ====================== Setup ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


# ====================== Main ======================
def main():
    # ---------------- Device check ------------------
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU for encoding.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU for encoding. Exiting program.{RESET}")
        exit()

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
    DATASET_NAME = 'UU_HDFS_train'
    DATASET = 'UU_HDFS'
    DATASETS_FOLDER = 'datasets'

    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}.csv'
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET_NAME}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET_NAME}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET_NAME}_All_pre_final_global_features.pkl'

    # ---------------- Initialize classes ----------------
    logdata_read_obj = logdata_read()
    features_extracting_obj = features_extracting()
    features_engineering_obj = features_engineering()
    anomaly_detection_obj = anomaly_detector()
    model_evaluation_obj = model_evaluation()
    utilities_obj = Utilities()

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(
        ALL_DATASET_CSV_PATH, DATASET
    )

    # ---------------- Process normal data ----------------
    print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
    final_train_with_test = utilities_obj.processing_data_portion(
        train_df, validate_df, test_df, df_features
    )

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    number_component, best_topic_number = features_extracting_obj.features_extracting_configuring_tuning(
        features_extracting_obj,
        DOC_TOPIC_DF_PATH,
        SENTIMENT_DF_PATH,
        DATASET_NAME,
        PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
        final_train_with_test
    )

    final_train_with_test = pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)

    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    sequences_df, X_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test,
        DATASET
    )

    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")
    X_train_normal_labelled = X_sequences_df[y_sequences_df == 0]
    X_train_all_data = X_sequences_df[y_sequences_df != 888]
    X_unlabeled_from_train = X_sequences_df[y_sequences_df == 999]

    labelled_df_from_train = sequences_df[sequences_df['Temp_label'] == 0]
    ground_truth_labeled_data_from_train = labelled_df_from_train['Label']

    labelled_df_from_train_all_data = sequences_df[sequences_df['Temp_label'] != 888]
    ground_truth_train_all_data = labelled_df_from_train_all_data['Label']

    unlabeled_df_from_train = sequences_df[sequences_df['Temp_label'] == 999]
    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train['Label']

    unlabeled_df_from_test = sequences_df[sequences_df['Temp_label'] == 888]
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test['Label']

    # ---------------- Novelty detection and label establishment ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    X_train, y_train, X_test, y_test_truth = features_engineering_obj.novelty_detection_label_establishment(
        sequences_df,
        X_train_normal_labelled,
        X_unlabeled_from_train,
        ground_truth_unlabeled_data_from_train
    )

    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    y_test_truth, y_test_pred = anomaly_detection_obj.anomaly_detector(
        X_train, y_train, X_test, y_test_truth
    )

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation(
        number_component, y_test_truth, y_test_pred, DATASET, X_test
    )


if __name__ == "__main__":
    main()
