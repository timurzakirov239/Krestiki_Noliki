from common import load_vivino

df = load_vivino()

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nWineType counts:")
print(df["WineType"].value_counts())

print("\nPrice describe:")
print(df["Price"].describe())

print("\nRating describe:")
print(df["Rating"].describe())

print("\nNumberOfRatings describe:")
print(df["NumberOfRatings"].describe())

print("\nN.V. share (Year_is_nv):")
print(df["Year_is_nv"].value_counts(normalize=True))

# квантильные пороги для классов цены
q33 = df["Price"].quantile(0.33)
q66 = df["Price"].quantile(0.66)
print(f"\nPrice quantiles: q33={q33:.2f}, q66={q66:.2f}")

# пример формирования классов
def price_class(p):
    if p <= q33: return "cheap"
    if p <= q66: return "mid"
    return "premium"

tmp = df["Price"].map(price_class)
print("\nPriceClass distribution (example):")
print(tmp.value_counts())
