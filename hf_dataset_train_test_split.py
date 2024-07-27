from datasets import load_dataset

# Combine 'question' and 'answer' into a single 'text' field
def combine_qa(local_dataset):
    local_dataset['text'] = f"User: {local_dataset['Question']}\nAssistant: {local_dataset['Answer']}"
    return local_dataset

####################################################################################
# main
####################################################################################
if __name__ == '__main__':
    # Download the dataset
    dataset = load_dataset("boricua/qna-ocp-4.15")

    # Combine Q&A into a single 'text' column and
    # keep the "train" dataset out of the DatasetDict
    dataset = dataset.map(combine_qa)['train']
    dataset.remove_columns(["ID"]) # remove the ID column (index from original dataset)

    # Split the dataset
    split_dataset=dataset.train_test_split(test_size=0.1) # 10% for test

    # Print some statistics
    print(f"Total examples: {len(dataset)}")
    print(f"Training examples: {split_dataset['train'].num_rows}")
    print(f"Test examples: {split_dataset['test'].num_rows}")

    # Save the datasets as Huffing Face format
    print("Saving train and test datasets saved to disk.")
    split_dataset['train'].save_to_disk("qna_ocp_train")
    split_dataset['test'].save_to_disk("qna_ocp_test")

    # Save the datasets in parquet formats
    split_dataset['train'].to_parquet("qna_ocp_train.parquet")
    split_dataset['test'].to_parquet("qna_ocp_test.parquet")


