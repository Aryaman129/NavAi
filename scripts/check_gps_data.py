import pandas as pd

print("=" * 70)
print("CAN Data (Car Speed):")
print("=" * 70)
can_df = pd.read_csv('data/comma2k19/processed_real/parsed/can_data.csv')
print(can_df.head())
print(f"\nColumns: {list(can_df.columns)}")
print(f"\nCar speed stats:")
print(can_df['car_speed_mps'].describe())
