
run_name=2017-04-24_Kleb_barcode
fastq_dir=/MDHS/Research/SysGen-Lab/MinION/2017-04-24_Kleb_barcode/fastq/1_raw
fast5_dir=/MDHS/Research/SysGen-Lab/MinION/2017-04-24_Kleb_barcode/raw_fast5
flowcell_kit=FLO-MIN106_SQK-LSK108

# run_name=2017-07-21_Singapore_Klebs_Sydney_Acinetobacter
# fastq_dir=/MDHS/Research/SysGen-Lab/MinION/2017-07-21_Singapore_Klebs_Sydney_Acinetobacter/basecalling/workspace
# fast5_dir=/MDHS/Research/SysGen-Lab/MinION/2017-07-21_Singapore_Klebs_Sydney_Acinetobacter/raw_fast5
# flowcell_kit=FLO-MIN106_SQK-LSK108

# run_name=2017-08-15_Sydney_Acinetobacter
# fastq_dir=/MDHS/Research/SysGen-Lab/MinION/2017-08-15_Sydney_Acinetobacter/basecalling/workspace
# fast5_dir=/MDHS/Research/SysGen-Lab/MinION/2017-08-15_Sydney_Acinetobacter/raw_fast5
# flowcell_kit=FLO-MIN107_SQK-LSK108

# run_name=2017-08-23_More_Kleb_barcode
# fastq_dir=/MDHS/Research/SysGen-Lab/MinION/2017-08-23_More_Kleb_barcode/basecalling/workspace
# fast5_dir=/MDHS/Research/SysGen-Lab/MinION/2017-08-23_More_Kleb_barcode/raw_fast5
# flowcell_kit=FLO-MIN107_SQK-LSK108



cd ~/nanopore-data/cnn_demultiplexer/training_sets
mkdir -p "$flowcell_kit"
porechop -i "$fastq_dir" -b "$flowcell_kit"/"$run_name" --require_two_barcodes --verbosity 3 --end_size 250 --no_split > "$flowcell_kit"/"$run_name"_porechop.out
python3 raw_training_data_from_porechop.py "$flowcell_kit"/"$run_name"_porechop.out "$fast5_dir" > "$flowcell_kit"/"$run_name"_raw_training_data
python3 select_training_samples.py "$flowcell_kit"/"$run_name"_raw_training_data "$flowcell_kit"/"$run_name"

# We are only interested in the Porechop output, so we can clean up the binned reads.
rm -r "$flowcell_kit"/"$run_name"
