import pyarrow.parquet as pq

file_path = 'data/comma2k19/processed_real/comma2k19_data.parquet'
parquet_file = pq.ParquetFile(file_path)

# Read first row group with just GPS columns
first_batch = parquet_file.read_row_group(0, columns=[
    'processed_log_gnss_live_gnss_qcom_t.npy',
    'processed_log_gnss_live_gnss_qcom_value.npy',
])

df = first_batch.to_pandas()

# Get first segment's GPS data
gps_t = df.iloc[0]['processed_log_gnss_live_gnss_qcom_t.npy']
gps_v = df.iloc[0]['processed_log_gnss_live_gnss_qcom_value.npy']

print("GPS Timestamps (first 3):")
print(gps_t[:3])
print(f"\nGPS Values structure (first value):")
print(f"Length: {len(gps_v[0])}")
print(f"Values: {gps_v[0]}")
print(f"\nAll values from first GPS sample:")
for i, val in enumerate(gps_v[0]):
    print(f"  Index {i}: {val}")
