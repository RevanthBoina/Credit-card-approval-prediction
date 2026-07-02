"""Model training entry point."""

from __future__ import annotations

import json

from extracted.Credit-card-approval-prediction.models.train import train_pipeline  # type: ignore


def main() -> None:
    metrics = train_pipeline()
    print(json.dumps(metrics.get("leaderboard", []), indent=2))
    print(f"Best model: {metrics.get('best_model')}")


if __name__ == "__main__":
    main()

