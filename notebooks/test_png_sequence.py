from dataset_png_sequence import MouthSequenceDataset

dataset = MouthSequenceDataset(
    root_dir="src/data/RVTALL/Processed_cut_data/kinect_processed",
    num_frames=16,
    image_size=96,
    grayscale=True
)

print("Number of samples:", len(dataset))
print("Labels:", dataset.label_to_idx)

x, y = dataset[0]
print("One sample shape:", x.shape)
print("One sample label index:", y)