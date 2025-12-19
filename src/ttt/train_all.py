from __future__ import annotations

from .training import train_level


def main():
    print("=== Обучение моделей крестики-нолики (с учителем) ===")
    for level in ["junior", "mid", "senior"]:
        print(f"\n--- Уровень: {level} ---")

        def cb(ep, tr, va):
            print(f"Эпоха {ep:>3}: train_acc={tr:.3f} | val_acc={va:.3f}")

        train_level(level, seed=42, progress_cb=cb, save=True)
        print(f"Сохранено: models/ttt_{level}.joblib")

    print("\nГотово.")


if __name__ == "__main__":
    main()
