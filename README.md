# CNN demultiplexer for Oxford Nanopore reads


## Preparing training data


#### Run Porechop

To prepare training data, we first need to know which reads have which barcodes. To do this, we run Porechop:
```
porechop -i "$fastq_dir" -b porechop_dir --require_two_barcodes --verbosity 3 --no_split > porechop.out
rm -r porechop_dir
```

By running Porechop with its highest verbosity (`--verbosity 3`), we produce an output with all of the information we need. 
Since we only need the output file, we can delete the binned reads after Porechop finishes.

The `--require_two_barcodes` option ensures that reads are only given a barcode bin if they have a good match on their start and end. This serves two purposes. First, it makes binning stringent, reducing the risk of misclassified reads. Second, allows us to use each read for training both the read start and read end CNNs, simplifying the process.

The `--no_split` option turns off Porechop's middle-adapter search and we use it here simply to save time. A chimeric read with a middle adapter is not a problem for our training set.



#### Extract signal data

Use the `porechop` subcommand to process the Porechop output into a file of raw training data:
```
cnn_demultiplexer porechop porechop.out /path/to/fast5_dir > raw_training_data
```

This will produce a tab-delimited file with the following columns:
* `Read_ID`: e.g. 3d873ba9-d55a-45a4-9ad5-db4f64135f11
* `Barcode_bin`: a number from 1-12
* `Barcode_distance_from_start`
* `Barcode_distance_from_end`
* `Start_read_signal`: signal from the start of the read. This is variable in length, because some read signals begin with a fair amount of open pore signal which will be trimmed off in the next step.
* `Middle_read_signal`: signal from the middle of the read (used as no-barcode training)
* `End_read_signal`: like the start read signal, this is variable in length.



#### Balancing the training data

```
cnn_demultiplexer balance raw_training_data balanced_training_data
```

This command finalises the training data. It produces two separate files, one for read starts and one for read ends. Each file only has two columns: the barcode label and the signal. It balances the training data by ensuring that each barcode has the same number of samples (necessarily limited to the number of samples for the least abundant barcode). No-barcode samples are included as well, using signal from the middle of reads.



## Training the neural network

```
cnn_demultiplexer train balanced_training_data_read_starts
cnn_demultiplexer train balanced_training_data_read_ends
```
