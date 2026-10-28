import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR, RANDOM_SEED


def main():
    path = PROCESSED_DIR / "metadata.csv"
    df = pd.read_csv(path, dtype={"machine_id": str})

    if "test_part" in df.columns:
        print("Test set is already split. Nothing changed.")
    else:
        test_idx = df.index[df.split == "test"]
        # Keep machine ID and label proportions equal in both halves
        strata = df.loc[test_idx, "machine_id"] + "_" + df.loc[test_idx, "label"].astype(str)
        dev_idx, final_idx = train_test_split(test_idx, test_size=0.5,
                                              random_state=RANDOM_SEED, stratify=strata)
        df["test_part"] = ""
        df.loc[dev_idx, "test_part"] = "dev"
        df.loc[final_idx, "test_part"] = "final"
        df.to_csv(path, index=False)
        print("Test set split into 'dev' and 'final' halves.")

    t = df[df.split == "test"]
    print(pd.crosstab([t.test_part, t.label], t.machine_id, margins=True))


if __name__ == "__main__":
    main()