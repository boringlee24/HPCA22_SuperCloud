# HPCA'22 Paper: AI-Enabling Workloads on Large-Scale GPU-Accelerated System: Characterization, Opportunities, and Implications

Production high-performance computing (HPC) systems are adopting and integrating GPUs into their design to accommodate artificial intelligence (AI), machine learning, and data visualization workloads. To aid with the design and operations of new and existing GPU-based large-scale systems, we provide a detailed characterization of system operations, job characteristics, user behavior, and trends on a contemporary GPU-accelerated production HPC system. Our insights indicate that the pre-mature phases in modern AI workflow take up significant GPU hours while underutilizing GPUs, which opens up the opportunity for a multi-tier system. Finally, we provide various potential recommendations and areas for future investment for system architects, operators, and users.

## Dependencies

We have used Python version 3.7 and Jupyter Notebook version 6.2. The required packages can be installed using
```shell
pip install -r requirements.txt
```

## Download data

The SuperCloud data is available for download from the Amazon Open Data Registry via the following bucket:
```shell
s3://mit-supercloud-dataset/2022-hpca/
```
There are three files available:

* *dcgm.csv* : This file contains the GPU utilization information for each GPU assigned to each job, including power consumption, resource usage.
* *scheduler_data.csv*:  This file contains the scheduler-level information about each job id, including start/end/wait time, user id, job type.
* *nvidia_smi.csv*: This file contains the time-series information captured by nvidia-smi, for about 2000 jobs. Data is recorded every 100ms.

To download, please first install the Amazon Web Service Command Line Interface AWS CLI [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

Once AWS CLI is installed, run the following command to list the availble files:
```shell
aws s3 ls s3://mit-supercloud-dataset/2022-hpca/ --no-sign-request
```
The *nvidia_smi.csv* file is 42GB. For demonstration purpose, you just need to download *dcgm.csv* and *scheduler_data.csv* to be able to run the provided notebook. Run the following command to store the data at the ```Downloads``` directory:
```shell
mkdir ~/Downloads/hpca22_supercloud
aws s3 cp s3://mit-supercloud-dataset/2022-hpca/dcgm.csv ~/Downloads/hpca22_supercloud --no-sign-request
aws s3 cp s3://mit-supercloud-dataset/2022-hpca/scheduler_data.csv ~/Downloads/hpca22_supercloud --no-sign-request
```

## Demonstration

We have provided a jupyter notebook file to demonstrate how to use the data to generate some of the figures in the paper. You can find the notebook [here](https://github.com/boringlee24/HPCA22_SuperCloud/blob/main/notebook.ipynb)

## Usage

Please feel free to email me any questions about the dataset.
My email: [li.baol@northeastern.edu](li.baol@northeastern.edu)

If you are interested in using the dataset, please cite this paper.
