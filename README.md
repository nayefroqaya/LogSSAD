# LogSSAD
LogSSAD: Semi-Supervised Anomaly Detection in Log Series

## 📌 Description
**LogSSAD** is a novel approach for log-based anomaly detection, i.e., textual event series.
Main steps:

(1) Leverage a diverse set of features.

(2) Using only normal data, we apply OC-SVM to generate pseudo-labels and estimate labels for unlabeled log sequences in the training set.

(3) Employ an ensemble learning framework to train the dataset.

We conducted empirical evaluations on two open-source datasets, HDFS and BGL, and achieved the following results:

LogSSAD attains the #highest F1 score on the HDFS dataset and a #competitive F1 score on the BGL dataset. Additionally, it demonstrates efficient runtime performance during both training and testing, achieving the #shortest runtime compared to baseline methods on both GPU and CPU.

---
## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System | Description                          | Time Span  | # Messages   | Data Size | Link |
|-----------------|--------------------------------------|------------|--------------|-----------|------|
| **HDFS**        | Hadoop Distributed File System log   | 38.7 hours | 11,175,629   | 1.47 GB   | [LogHub](https://github.com/logpai/loghub) |
| **BGL**         | Blue Gene/L supercomputer log        | 214.7 days | 4,747,963    | 708.76 MB | [LogHub](https://github.com/logpai/loghub)  |
---
## ⚙️ Environment
All libraries are specified with their versions in the requirements file.

---
## 🛠️ Preparation
Steps to run LogSSAD:

1. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`).
2. In main.py file, make the name of the dataset BGL or HDFS.
3. Be sure you install all libraries in requirement file.
---
## 📌 Data Parsing
1. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
2. On GitHub, only a subset of the dataset is used. You should use the full datasets (HDFS and BGL).
   For new dataset: 
1. Place name of the dataset in main.py file.  
2. Implement a new case for new dataset in the LogdataRead class (see `src/logdata_reader.py`).
---
## 🚨 Anomaly Detection
To apply LogSSAD on log data:
* run the main fucntion (see `src/main.py`)
---
## 📬 Contact
We are happy to answer your questions:   

| Name               | Email Address                             |
|--------------------|-------------------------------------------|
| Nayef Roqaya       | roqaya@staff.uni-marburg.de               |
| Thorsten Papenbrock| papenbrock@informatik.uni-marburg.de      |
| Hajira Jabeen      | hajira.jabeen@uk-koeln.de                 |

## 📬 Citation
@inproceedings{roqaya2025logad,
  title={LogSSAD: Semi-Supervised Anomaly Detection in Log Series},
  author={Roqaya, Nayef and Papenbrock, Thorsten and Jabeen, Hajira},
  booktitle={Proceedings of the ACM/SIAM International Conference on Data Mining (SDM)},
  year={2025},
  publisher={EDBT26}
}




   
