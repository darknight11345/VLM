import sys
import os

# Add the path to your project if needed so Python can find data.py and dependencies
sys.path.append("/pfs/work9/workspace/scratch/ul_swv79-pixtral/Pixtral-Finetune/src/training")

from data import LazySupervisedDataset, DataArguments  # import your classes

def main():
    # Create dummy DataArguments object with necessary attributes
    data_args = DataArguments(
        data_path="/pfs/work9/workspace/scratch/ul_swv79-pixtral/Dataset/Training_dataset",
        image_folder="/pfs/work9/workspace/scratch/ul_swv79-pixtral/Dataset/Training_dataset/image_dots",
        qa_json_path="/pfs/work9/workspace/scratch/ul_swv79-pixtral/Dataset/Training_dataset/qa_dots.json",
        lazy_preprocess=True
    )

    # You also need to create or load your processor (like a tokenizer or feature extractor)
    # This depends on what processor you use in your training
    # Here's an example, adapt as needed:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")  # or your actual processor

    dataset = LazySupervisedDataset(data_path=data_args.data_path, processor=processor, data_args=data_args)
    print("Dataset length:", len(dataset))

    # Load first sample to test __getitem__
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Sample input_ids shape:", sample['input_ids'].shape)
    if sample.get("pixel_values") is not None:
        print("Sample pixel_values shape:", sample['pixel_values'].shape)

if __name__ == "__main__":
    main()
