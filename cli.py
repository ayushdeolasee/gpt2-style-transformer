import typer
import train

app = typer.Typer()

@app.command()
def train_cli(
    dataset_location: str = "fineweb10B",
    epochs: int = 19073,
    batch_size: int = 4,
    block_size: int = 1024,
    total_batch_size: int = 524288,
    lr: float = 3e-4,
):
    train.train(
        dataset_location=dataset_location,
        epochs=epochs,
        batch_size=batch_size,
        block_size=block_size,
        total_batch_size=total_batch_size,
        lr=lr,
    )

if __name__ == "__main__":
    app()
