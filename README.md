# Required python packages
- python 3.8.*
- torch 1.12.1 with cuda
- numpy
- pandas
- tqdm

# Code Run Instructions

> Pretrained saved BERT model location: 
> __pretrained_model/bert_pretrained_model__

## To run a fine tuning task, run the following command
> python main.py -pretrained_bert_checkpoint pretrained_model/bert_pretrained_model

You can pass the following parameters while performing a fine tune task.

- -train_dataset_path 
- -train_label_path 
- -test_dataset_path 
- -test_label_path

Current value of these parameters are set to:

- -train_dataset_path : data/sample_train_data.txt
- -train_label_path : data/sample_train_label.txt
- -test_dataset_path : data/sample_test_data.txt
- -test_label_path : data/sample_test_label.txt

The test data is used for validation purpose only.

> All the logs are saved inside the ___log___ folder automatically after the fine tune task is completed.
> 
> The fine tuned model is saved inside the ___output___ folder.
>
> Results have been derived from the logs.
> 
> The ___fine tuned model___ can be used for predicting correctness of the final answer after completing the provided optional tasks.