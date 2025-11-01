# Data Directory

This directory is for datasets used in your reproducibility study.

## Important Notes

⚠️ **Do not commit large datasets to Git**
- Add large files to `.gitignore`
- Provide download scripts or links instead
- Consider using Git LFS for medium-sized files

## Organization

### raw/
Original, unmodified data
- Keep pristine copies of downloaded data
- Document data sources and versions
- Include checksums for verification

### processed/
Preprocessed data ready for experiments
- Document all preprocessing steps
- Include scripts that generate processed data from raw data

### metadata/
Information about the datasets
- Data dictionaries
- Statistics and summaries
- Provenance information

## Example Structure

```
data/
├── README.md
├── download_data.sh
├── checksums.txt
├── raw/
│   ├── README.md
│   └── [raw data files or download instructions]
├── processed/
│   ├── README.md
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── metadata/
    ├── data_dictionary.md
    ├── statistics.json
    └── provenance.md
```

## Data Access

### Original Paper Data

**Source:** [URL or citation]

**Access Method:** [How to obtain the data]

**License:** [Data license information]

**Version:** [Dataset version used]

### Download Instructions

```bash
# Example download commands
# cd data/raw
# wget [URL]
# or
# python download_data.py
```

### Checksums

Verify data integrity using checksums:

```bash
# Example
# sha256sum -c checksums.txt
```

| File | SHA256 |
|------|--------|
|      |        |

## Preprocessing

Document all preprocessing steps:

1. [Step 1]
2. [Step 2]
3. [Step 3]

Run preprocessing:

```bash
# Example
# python ../code/scripts/preprocess_data.py
```

## Data Statistics

### Raw Data

| Dataset | Samples | Features | Size |
|---------|---------|----------|------|
|         |         |          |      |

### Processed Data

| Split | Samples | Features | Size |
|-------|---------|----------|------|
| Train |         |          |      |
| Val   |         |          |      |
| Test  |         |          |      |

## Ethical Considerations

[Document any ethical considerations regarding data use]

## Privacy and Compliance

[Document any privacy or compliance requirements]
