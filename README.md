This is project of using pypi libarary segmentation_models_pytorch, briefly called SMP, to segment a target object such as a car or bycical.
SMP includes multiple well-known models. The default model used in this project is resent34 and related weights is imagenet
We only trained single class in this project as an example reference.

train.py: used to train SMP model under configuration config.py where training dataset called CamVid
valid.py: used to validate the trained model with a video cliip called racing_car.sd.mp4


# Prerequisites

## Connections
-   Generate a ssh key pair on your own computer, then paste the **public key** to the [Digital Research Alliance of Canada (The Alliance)](https://ccdb.alliancecan.ca/ssh_authorized_keys).

-   Login to the Alliance from terminal: ```ssh -Y USERNAME@SERVERNAME.computecanada.ca```

    e.g. ```ssh -Y USERNAME@beluga.computecanada.ca```

-   Generate a ssh key pair on the Alliance, then paste the public key to your [Github](https://github.com/settings/keys). Then you can access your Github on the Alliance. ```ssh-keygen -t ed25519``` or simply ```ssh-keygen```

-   Create a symbolic link on the home directory that associates to your project on the Alliance (*optional*): ```ln -s <target_file_or_directory> <link_name>```

    e.g. ```ln -s projects/def-jyzhao/zzhou186/prj_smp/ prj_smp_workspace```

    Next time we can ```cd prj_smp_workspace``` instead of ```cd projects/def-jyzhao/zzhou186/prj_smp```

## Virtual Environment On The Alliance
-   Discover the versions of Python available: ```module avail python```

-   Load a Python module (***need to be done outside of the virtual environment***): e.g. ```module load python/3.12.4```

-   Load opencv-python module (***need to be done outside of the virtual environment***), 4.10.0 is the latest version available on the Alliance by the time of writing (cannot be installed using pip): ```module load opencv/4.10.0```

-   Create a virtual environment **inside your project:** ```virtualenv --no-download env```

-   Activate the virtual environment: ```source env/bin/activate```

-   Upgrade pip in the virtual environment: ```pip install --no-index --upgrade``` (--no-index is to download the available packages from the Alliance instead of from PyPI)

-   Install the packages listed on the requirement.txt: ```pip install --no-index -r requirement.txt```

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