import pyarrow.parquet as pq

file_path = 'data/comma2k19/processed_real/comma2k19_data.parquet'
parquet_file = pq.ParquetFile(file_path)

# Read first row group with CAN speed
first_batch = parquet_file.read_row_group(0, columns=[
    'processed_log_can_speed_t.npy',
    'processed_log_can_speed_value.npy',
])

df = first_batch.to_pandas()

# Get first segment's speed data
speed_t = df.iloc[0]['processed_log_can_speed_t.npy']
speed_v = df.iloc[0]['processed_log_can_speed_value.npy']

print("CAN Speed Timestamps (first 5):")
print(speed_t[:5])
print(f"\nCAN Speed Values (first 5):")
for i, val in enumerate(speed_v[:5]):
    print(f"  {i}: {val} (type: {type(val).__name__})")
