from src.train import train_and_persist


if __name__ == "__main__":
    metadata = train_and_persist()
    print(f"Trained best model: {metadata['model_name']}")
