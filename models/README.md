# Deepbinner pre-trained models

### 1D ligation barcoding

The `EXP-NBD103_read_starts` and `EXP-NBD103_read_ends` models were trained on [SQK-LSK108 1D ligation](https://store.nanoporetech.com/ligation-sequencing-kit-1d.html) reads using the [EXP-NBD103 native barcoding expansion](https://store.nanoporetech.com/native-barcoding-expansion-1-12.html). Both R9.4 and R9.5 flowcells were included in the training set, so these models should work well with both.

Since native barcoding adds barcodes to both the start and end of reads, there are two models. Use them like this:
```
deepbinner classify -s EXP-NBD103_read_starts -e EXP-NBD103_read_ends input
```

Or with the preset:
```
deepbinner classify --native input
```



### Rapid barcoding

The `SQK-RBK004_read_starts` model was trained on [SQK-RBK004 rapid barcoding](https://store.nanoporetech.com/rapid-barcoding-kit.html) reads. Many thanks to [Matthew Croxen](https://twitter.com/m_croxen) for sharing them with me to train this model. The training reads were sequenced on R9.4 flowcells, so I'm not sure how well this model will perform on reads from R9.5 flowcells.

Since rapid barcoding only adds a barcode to the start of reads, there is only one model for this kit. Use it like this:
```
deepbinner classify -s SQK-RBK004_read_starts input
```

Or with the preset:
```
deepbinner classify --rapid input
```



### Sharing reads

I am lacking some types of reads in my training sets, so if you can share yours with me, I'd be very grateful! In particular, I would like:
* Native barcoding reads using the newer [SQK-LSK109 kit](https://store.nanoporetech.com/ligation-sequencing-kit.html).
* [SQK-RBK004 rapid barcoding](https://store.nanoporetech.com/rapid-barcoding-kit.html) reads from R9.5 flowcells (to make the `SQK-RBK004_read_starts` model more robust)
* SQK-RBK001 rapid barcoding reads (the older kit, predecessor to SQK-RBK004)

If you have any to share, please [let me know](https://github.com/rrwick/Deepbinner/issues/new)! A caveat: the reads should be from whole genome sequencing runs of bacterial genomes or larger. I'm concerned that if the reads came from amplicons or small (e.g. viral) genomes, then the neural network might learn to recognise the sequenced material in addition to the barcode – not what we want! Random whole genome sequencing ensures that the sequenced material is mostly different in each read, so the network must learn to recognise the _barcodes_.

A Deepbinner model for PCR-barcoded reads may be possible, but the caveat above especially applies here. I suspect to make it work, I would need reads from a lot of separate PCR-barcoded runs, each with unique amplicons.

