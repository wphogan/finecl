'''
Script to:
- generate list of high-quality UIDs
- generate training file that only contains high-quality labels

Run Order:
1. Run stage 1 on all data.
2. Run this file to collect first-learned UIDs per first N epochs and save new 'high-quality' training data file
3. Run stage 1 on new high-quality data
'''
import json
import os.path
from src.stage_1.utils import JsonlReader, target_directory
from datetime import datetime


# mode=create_high_qual_training_set.yaml
def create_high_qual_training_set(config):
    # Pre
    begin_time = datetime.now()
    print(f'Began script: {begin_time}')

    # 0. Construct directory path
    '''Load data from log/experiments/stage_1_experiment/--DATETIME_STRING--/first_learned/train_distant_fl'''
    '''e.g. --------> log/experiments/stage_1_experiment/2021-12-10_11-39-41/first_learned/train_distant_fl'''
    # datetime_str = '2022-01-04_11-35-41_BL' # <--- original batch-based Baseline CE
    datetime_str = '2022-01-26_18-42-27_BL_whole'  # <--- whole-training set eval Baseline CE
    split_type = 'train_distant'
    target_dir, _ = target_directory(config.proj_root_dir, datetime_str)
    dir_train_fl = os.path.join(target_dir, 'first_learned', f'{split_type}_fl')

    # 3. Load original training data
    fname_orig_train_data = os.path.join(config.dir_preprocessed, f'{split_type}.json')
    orig_data = json.load(open(fname_orig_train_data))
    print(f'Loaded file: {fname_orig_train_data} containing {len(orig_data)} documents.')

    # 1. Open JSONL files, load first_learned from first N epochs
    keep_first_n_epochs_multirun = [1, 2, 3, 4, 5, 6]
    for keep_first_n_epochs in keep_first_n_epochs_multirun:
        print(f'Generating data for {keep_first_n_epochs} epoch')
        high_quality_uids = set()
        for epoch in range(keep_first_n_epochs):
            # file_name = f'_train_{epoch:03}.json'
            file_name = f'train_distant_{epoch:03}_whole.json'
            file_name_full = os.path.join(dir_train_fl, file_name)  # add full path to fname
            file_data = list(iter(JsonlReader(file_name_full)))[0]  # get data in file

            # 1b. Compile set of high-quality UIDs from first N epochs
            uids = set(file_data[str(epoch)])
            high_quality_uids.update(uids)
            print(
                f'Read file: {file_name}, epoch HQ UID count: {len(uids)}, cumulative UID count: {len(high_quality_uids)}')

        # 2. Save high-quality set of UIDs
        fname_out = f'high_quality_uids_n{keep_first_n_epochs}.json'
        fname_out = os.path.join(dir_train_fl, fname_out)  # add full path to fname
        with open(fname_out, "w") as wf:
            wf.write(json.dumps(list(high_quality_uids)))
            print(f'Wrote file: {fname_out} with {len(high_quality_uids)} high-quality UIDs.')

        # 4. Trim data to only contain high-quality UIDs
        new_data = []
        total_labels, total_labels_hq = 0, 0

        for document in orig_data:
            high_quality_doc_labels = []
            total_labels += len(document['labels'])
            for label in document['labels']:
                if label['uid'] in high_quality_uids:
                    high_quality_doc_labels.append(label)

            document['labels'] = high_quality_doc_labels  # doc keeps only high-quality labels
            new_data.append(document)  # this doc contains at least one high-quality label. Save it!
            total_labels_hq += len(high_quality_doc_labels)

        # 5. Save new high-quality training data
        fname_out = f'{split_type}_high_quality_n{keep_first_n_epochs}.json'
        fname_out = os.path.join(dir_train_fl, fname_out)  # add full path to fname
        with open(fname_out, "w") as wf:
            wf.write(json.dumps(new_data))
            print(f'Wrote file: {fname_out} containing {len(new_data)} documents.')
            print(f'N Orig Labels: {total_labels} || N High-Qual Labels: {total_labels_hq}')
            print(f'Ratio --> High-Qual Labels/All Labels: {total_labels_hq / total_labels}')
        print('Clearing variables.')
        del high_quality_uids, high_quality_doc_labels, new_data

    end_time = datetime.now() - begin_time
    print(f'Began script: {begin_time}')
    print(f'------Time to complete script: {end_time}------')
    return
