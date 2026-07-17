import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    """
    Runs the full pipeline: ingest raw data -> split & save train/test csv ->
    transform (preprocess + save preprocessor.pkl) -> train models
    (pick the best one + save model.pkl).
    """
    try:
        logging.info("Training pipeline started")

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        model_trainer = ModelTrainer()
        best_model_name, accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline completed. Best model: {best_model_name}, accuracy: {accuracy:.4f}")
        return best_model_name, accuracy

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    best_model_name, accuracy = run_training_pipeline()
    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {accuracy:.4f}")
