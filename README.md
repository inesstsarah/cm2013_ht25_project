This is a repository of the final Signal Processing Project, developed and maintained by Group 13 with the following members Hao Tang, Filip Stenlund, and Ines Siti Sarah.

To set up the environment, run the following commands in a command prompt terminal with Anaconda installed and configured from the root directory. 

````

cd Python
conda create -n "signals-env" python=3.11.13
conda activate signals-env
conda install pip
pip install -r requirements.txt


````
Make sure that all data files are within the data/ folder before continuing. 
To conduct testing:

````

cd Python
pytest


````
To run the pipeline:
````

cd Python
python main.py


````

To run inference on holdout files, first make sure that all holdout files are in the data/holdout/ folder. Then run the following commands.
````

cd Python
python run_inference.py

````

