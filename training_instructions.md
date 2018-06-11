# Training the Deepbinner CNN

These instructions cover all the steps necessary to train CNN models for use in the `deepbinner classify` command.


## Prerequisites

Lots of barcoded reads! More training data is always better.


## Basecall with Albacore

```
read_fast5_basecaller.py -f FLO-MIN106 -k SQK-LSK108 -i /path/to/fast5_dir -t 16 -s basecalling -o fastq --disable_filtering --barcoding --recursive
```



## Run Porechop

To prepare training data, we first need to know which reads have which barcodes. To do this, we run Porechop:
```
porechop -i basecalling/workspace -b porechop_dir --require_two_barcodes --verbosity 3 --no_split > porechop.out
rm -r porechop_dir
```

By running Porechop with its highest verbosity (`--verbosity 3`), we produce an output with all of the information we need. CNN demultiplexer will read the Porechop output file, so we can delete the binned reads after it finishes.

The `--require_two_barcodes` option ensures that reads are only given a barcode bin if they have a good match on their start and end. This serves two purposes. First, it makes binning stringent, reducing the risk of misclassified reads. Second, allows us to use each read for training both the read start and read end CNNs, simplifying the process. __Do not use this option if you are preparing rapid barcoding data, as there are no end-barcodes for rapid reads.__

The `--no_split` option turns off Porechop's middle-adapter search and we use it here simply to save time. A chimeric read with a middle adapter is not a problem for our training set.



## Extract signal data

Use the `porechop` subcommand to process the Porechop output into a file of raw training data:
```
deepbinner porechop porechop.out /path/to/fast5_dir > raw_training_data
```

This will produce a tab-delimited file with the following columns:
* `Read_ID`: e.g. 3d873ba9-d55a-45a4-9ad5-db4f64135f11
* `Barcode_bin`: a number from 1-12
* `Start_read_signal`: signal from the start of the read. This is variable in length, because some read signals begin with a fair amount of open pore signal which will be trimmed off in the next step.
* `Middle_read_signal`: signal from the middle of the read (used for training samples without a barcode)
* `End_read_signal`: signal from the end of the read. Like the start read signal, this is variable in length.



## Balancing the training data

This command finalises the training data:
```
deepbinner balance raw_training_data training
```

It balances the data by ensuring that each barcode has the same number of samples (necessarily limited to the number of samples for the least abundant barcode). No-barcode samples are included as well, using signal from the middle of reads and randomly generated signals.

This command produces two separate files, one for read starts and one for read ends. Each file only has two columns: the barcode label and the signal. If you are training on rapid reads (which only have a start barcode), you can delete the end barcode file.

It also attempts to trim off open-pore signal at the start of the signal, so it outputs data from when the real read has begun. However, this process isn't perfect (more on that in the refining step below).



## Training the neural network

The following instructions assume you're training a model that has barcodes on both the start and end of reads. If you have a start-barcode-only dataset (like rapid barcoding reads), just ignore the end-read commands.

Now it's time to actually train the CNN!

```
deepbinner train training_read_starts read_start_model
deepbinner train training_read_ends read_end_model
```

This part can be quite time consuming, and so a big GPU is definitely recommended! Even with a big GPU, it may take many hours to finish.

Options to change some parameters:

* `--signal_size`
* `--barcode_count`
* `--epochs`: Too few and the network won't get good enough. Too many and it's a waste of time.
* `--batch_size`: Larger values may work better but will use more memory.
* `--test_fraction`: What fraction of the data will be set aside for use as a validation set. If you have no interest in assessing the model, set this to 0.0 to use all of your data for training.

If you would like to [design your own CNN architecture](https://keras.io/getting-started/functional-api-guide/), you'll need to code it up yourself by modifying the `build_network` function in `network_architecture.py`.



## Refining the training set and retraining

The training data we prepared earlier may have some duds in it. This may be due to incorrect trimming of open pore signal. Or it may be that Porechop produced a false positive. Either way, our training data may have a signal which is supposed to have a particular barcode but does not. This is not good for training!

To remedy this, we use our trained CNN to classify our training data:
```
deepbinner classify -s read_start_model training_read_starts > read_starts_classification
deepbinner classify -e read_end_model training_read_ends > read_ends_classification
```

As you would expect, this process should classify most samples to the same barcode bin as their training label. The small fraction that do not match should be excluded from the training set. These commands will produce new training sets with those samples filtered out:
```
deepbinner refine training_read_starts read_starts_classification > training_read_starts_refined
deepbinner refine training_read_ends read_ends_classification > training_read_ends_refined
```

Now that a new, better training set is available, we can retrain our CNN and hopefully produce even better models than before:
```
rm read_start_model read_end_model
deepbinner train training_read_starts_refined read_start_model
deepbinner train training_read_ends_refined read_end_model
```


## Refining the training set and retraining (in a loop)

There may be benefit in repeating the previously described method for refining the training data. This is because each time it runs, the program will randomly allocate a the samples to training and validation sets (default 90:10 ratio). Samples in the training set can be 'memorised' by the network, i.e. even if they are bogus samples, the network may be able to classify them correctly. Refining the training data is therefore most effective for samples that were in the validation set, where memorisation wasn't possible.

You may therefore want to repeat the refining process multiple times, so any bogus sample has a greater chance of landing in the validation set and then being culled. Here is a Bash loop I used to do this:
```
training_data="training_read_starts"
model="read_start_model"
classifications="read_starts_classification"

printf "Starting training data count: "
wc -l "$training_data"

for i in {1..10}; do
    rm -f "$model"
    deepbinner train --epochs 25 "$training_data" "$model"
    deepbinner classify -s "$model" "$training_data" > "$classifications"
    deepbinner refine "$training_data" "$classifications" > $training_data"_refined"
    rm "$training_data"
    mv $training_data"_refined" "$training_data"
    printf "Iteration "$i" training data count: "
    wc -l "$training_data"
done
```

