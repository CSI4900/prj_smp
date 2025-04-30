# Overview
This project uses a PyTorch library called Segmentation Models Pytorch (SMP). We used the U-Net architecture with a MobileOne_s4 encoder pre-trained on ImageNet. This optimal combination was identified in [priori work](CSI4900_Dageng-Ding.pdf) by a student, Dageng Ding.

We trained three models of this setup using the [Cityscapes Foggy dataset](https://www.cityscapes-dataset.com/file-handling/?packageID=29) (leftImg8bit_trainvaltest_foggy.zip),  and labeled them SMP #1, SMP #2, and SMP #3, in the order they were trained.

| Trained Model | Epochs | Classes | Training Images | Validating Images | Testing Images
|----------|----------|----------|-----------|------------|------------|
| SMP #1   | 30  |  30  |  462 images  (City: Cologne) | 177 images (City: Lindau) | 138 images (City: Bonn)
| SMP #2   | 50  |  12  | 2220 images (City: Aachen, Bochurn, Bremen, and Cologne)  | 699 images (City: Lindau, Munster) | 681 images (City: Bielefeld, Bonn)
| SMP #3   | 50  |  30  | 2220 images (City: Aachen, Bochurn, Bremen, and Cologne)  | 699 images (City: Lindau, Munster) | 681 images (City: Bielefeld, Bonn)

Below are the brief descriptions for each folder and script:
- train.py: This script is used to train the SMP model under the configuration of `config.py`
- valid.py: This script is used to validate the trained model with a video clip
- graphs.py: This script is used to generate the Dice Loss and IoU graph using the training log file
- config.py: This is a configuration file for this project
- dataset: This folder contains configuration files for the training dataset
- params: This folder is used to store the trained model when running `train.py`
- results: This folder is used to store the model output when running `valid.py`
- test_data: This folder is used to store the video clips for model testing or validation
- training_data: This folder contains the training dataset

The following scripts only work for Compute Canada:
- train_job.sh: This script is used to submit a job to run `train.py` on Compute Canada
- valid_job.sh: This script is used to submit a job to run `valid.py` on Compute Canada
- load_modules.sh: This script is used to load relative modules on Compute Canada


# Prerequisites

## Connections
-   Generate a ssh key pair on your own computer, then paste the **public key** to the [Digital Research Alliance of Canada (The Alliance)](https://ccdb.alliancecan.ca/ssh_authorized_keys).

-   Login to the Alliance from terminal: ```ssh -Y USERNAME@SERVERNAME.computecanada.ca```

    e.g. ```ssh -Y USERNAME@beluga.computecanada.ca```

-   Generate a ssh key pair on the Alliance, then paste the public key to your [Github](https://github.com/settings/keys). Then you can access your Github on the Alliance. ```ssh-keygen -t ed25519``` or simply ```ssh-keygen```

-   Create a symbolic link on the home directory that associates to your project on the Alliance (*optional*): ```ln -s <target_file_or_directory> <link_name>```

    e.g. ```ln -s projects/def-jyzhao/USERNAME/prj_smp/ prj_smp_workspace```

    Next time we can ```cd prj_smp_workspace``` instead of ```cd projects/def-jyzhao/USERNAME/prj_smp```

## Virtual Environment On The Alliance
-   Discover the versions of Python available: ```module avail python```

-   Load a Python module (***need to be done outside of the virtual environment***): e.g. ```module load python/3.12.4```

-   Load opencv-python module (***need to be done outside of the virtual environment***), 4.10.0 is the latest version available on the Alliance by the time of writing (cannot be installed using pip): ```module load opencv/4.10.0```

-   Load Git LFS (large File Storage) (***need to be done outside of the virtual environment***): ```module load git-lfs```

-   Create a virtual environment **inside your project:** ```virtualenv --no-download env```

-   Activate the virtual environment: ```source env/bin/activate```

-   Initializes Git Large File Storage (LFS): ```git lfs install```

-   Upgrade pip in the virtual environment: ```pip install --no-index --upgrade pip``` (--no-index is to download the available packages from the Alliance instead of from PyPI)

-   Install the packages listed on the requirements.txt: ```pip install --no-index -r requirements.txt```

-   Exit the virtual environment: ```deactivate```

## Job Scheduling

-   Create a script file, e.g. train_job.sh

-   Submit a job: ```sbatch train_job.sh```

-   Cancel a job: ```scancel YOUR_JOBID```

-   Check the submitted jobs' information: ```sq```

# References

1. The Alliance wiki page: [Using SSH keys in Linux](https://docs.alliancecan.ca/wiki/Using_SSH_keys_in_Linux)

2. The Alliance wiki page: [Python](https://docs.alliancecan.ca/wiki/Python)

3. The Alliance wiki page: [Running jobs](https://docs.alliancecan.ca/wiki/Running_jobs)

4. 

Version: Feb 6, 2025 00:32
