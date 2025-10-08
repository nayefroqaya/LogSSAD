# **LogSSAD**
This is the basic implementation of our submission in EDBT 2026. LogSSAD: Semi-Supervised Anomaly Detection in Log Series

## 📌 Description
LogSSAD is a novel approach for log-based anomaly detection, i.e., textual event series. We conducted empirical evaluations on two open-source datasets, HDFS and BGL, and achieved the following results:  
LogSSAD attains the highest F1 score on the HDFS dataset and a competitive F1 score on the BGL dataset. Additionally, it demonstrates efficient runtime performance during both training and testing, achieving the shortest runtime compared to baseline methods on both GPU and CPU.

---

## Project Structure
<pre>
├─ datasets/               # Main entry point for PLELog datasets  
├─ drain_parser/           # Configuration and parser scripts for Drain  
├─ src/  
│  ├─ main.py              # Main script to trigger the full pipeline  
│  ├─ logdata_read.py      # Reads parsed data and performs cleaning and column selection  
│  ├─ features_extracting.py  # Extracts features from log events  
│  ├─ features_engineering.py # Performs feature transformation  
│  ├─ anomaly_detection.py    # Contains anomaly detection modules  
│  ├─ model_evaluation.py     # Evaluation module (Precision, Recall, F1-score, etc.)  
│  └─ utility.py              # Helper functions used across different stages  
</pre>   
---

## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System | Description                          | Time Span  | # Messages   | Data Size | Link |
|-----------------|--------------------------------------|------------|--------------|-----------|------|
| HDFS            | Hadoop Distributed File System log   | 38.7 hours | 11,175,629   | 1.47 GB   | [LogHub](https://github.com/logpai/loghub) |
| BGL             | Blue Gene/L supercomputer log        | 214.7 days | 4,747,963    | 708.76 MB | [LogHub](https://github.com/logpai/loghub)  |

---

## ⚙️ Environment
All libraries are specified with their versions in the requirements file (e.g., LogSSAD/requirements.txt).

---

## 🛠️ Preparation
Steps to run LogSSAD:

1. Install all required libraries from the requirements file (e.g., LogSSAD/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name to either BGL or HDFS.
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is avaible for BGL and HDFS. Uncomment the lines of the dataset you need to use

---

## 📌 Data Parsing
1. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
2. To start the parsing process, run (drain_parser/demo.py). 
3. The parsing output will be generated and saved in the datasets directory.

---

## 🚨 Anomaly Detection 
To apply the LogSSAD pipeline on log data:
* Run the main function (`src/main.py`).
* The main function executes all stages as one pipeline: data preprocessing, anomaly detection, and evaluation.

---

## 📬 Contact
We are happy to answer your questions:   

| Name               | Email Address                             |
|--------------------|-------------------------------------------|
| Nayef Roqaya       | roqaya@staff.uni-marburg.de               |
| Thorsten Papenbrock| papenbrock@informatik.uni-marburg.de      |
| Hajira Jabeen      | hajira.jabeen@uk-koeln.de                 |

---

## 📬 Citation
```bibtex
@inproceedings{roqaya2025LogSSAD,
  title={LogSSAD: Semi-Supervised Anomaly Detection in Log Series},
  author={Roqaya, Nayef and Papenbrock, Thorsten and Jabeen, Hajira},
  booktitle={Proceedings of annual International Conference on Extending Database Technology (EDBT)},
  year={2026},
  publisher={EDBT26}
}
