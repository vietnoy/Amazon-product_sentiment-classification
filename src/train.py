from data_preprocessing import load_and_preprocess
from model import train_model
import os

def main():
    print("Loading and preprocessing data...")
    train_df, val_df, _ = load_and_preprocess()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    model_dir = os.path.join("saved_model", "distilbert_model")
    os.makedirs(model_dir, exist_ok=True)

    print("Training model...")
    model, tokenizer = train_model(train_df, val_df, output_dir=model_dir)

    print("Model training complete. Saved to:", model_dir)

if __name__ == "__main__":
    main()
