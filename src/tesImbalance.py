import pandas as pd
import numpy as np
import config as C

df = pd.read_csv(C.KURS_CLEAN_CSV)

df = df.dropna(subset=["log_return", "direction"])

print("=" * 60)
print("DISTRIBUSI KELAS TARGET")
print("=" * 60)

counts = df["direction"].value_counts()
percentages = df["direction"].value_counts(normalize=True) * 100

summary = pd.DataFrame({
    "jumlah": counts,
    "persentase": percentages
})

print(summary.to_string())

print("\n" + "=" * 60)
print("STATISTIK LOG RETURN PER KELAS")
print("=" * 60)

print(
    df.groupby("direction")["log_return"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .to_string()
)

print("\n" + "=" * 60)
print("PEMERIKSAAN THRESHOLD")
print("=" * 60)

print(f"FLAT_THRESHOLD : {C.FLAT_THRESHOLD}")
print(f"Batas atas     : {C.FLAT_THRESHOLD * 100:.4f}%")
print(f"Batas bawah    : {-C.FLAT_THRESHOLD * 100:.4f}%")