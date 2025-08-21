# watershed_prep
grass GIS watershed processing notebooks. To be used in conjunction with dem_prep repo as needed

# Introduction 


# Build and Test
The notebooks executes a script in the src folder and the notebook runs in a Docker container. This requires that the user install

* Docker desktop - https://www.docker.com/products/docker-desktop
* Windows sub-Linux SYstem - https://docs.microsoft.com/en-us/windows/wsl/install-win10

Note that this only requires completely steps up to and including Step 5. No need to install a Linux distribution.

## Docker Container Build

### Running the Container

To run the container, open the command line prompt and 
- Change directory to the project folder with the docker-compose.yml file. 
- Run the command 'docker-compose up --build' to run the docker container.

Before running the container, open the docker-compose.yml file and  examine if folder volumes on the container map to the correct local folders on one's computer. 
Note under the 'volumes' section of the docker-compose file, the folder location on the local computer is given before the colon while the location in the container is given after the colon. Just change thte location on the computer (text before the colon), as desired. Also note the port over which the Jupyter notebook is accessed (here it set as 8080). Change as necessary.

Once the container is running, one uses Chrome or another web browser to work with the code through one of three options:  
* Jupyter Lab -- http://localhost:8080/lab
* Jupyter Notebook -- http://localhost:8080
* Nteract Notebook -- http://localhost:8080/nteract

For more information on Nteract, please refer to https://nteract.io/. Nteract was developed by Netflix https://netflixtechblog.com/notebook-innovation-591ee3221233


### Altering the Docker Container
The Docker container allows for multiple users to consistently setup and run the notebook and setup of the container is outlined in the Dockerfile (in the Docker folder). The Dockerfile starts with a base image pulled from the Jupyter Docker Stacks:

https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html

For this project, the container is built upon the jupyter/scipy-notebook image, which contains

* Everything in jupyter/minimal-notebook and its ancestor images
* dask, pandas, numexpr, matplotlib, scipy, seaborn, scikit-learn, scikit-image, sympy, cython, patsy, statsmodel, cloudpickle, dill, numba, bokeh, sqlalchemy, hdf5, vincent, beautifulsoup, protobuf, xlrd, bottleneck, and pytables packages
* ipywidgets and ipympl for interactive visualizations and plots in Python notebooks
* Facets for visualizing machine learning datasets.

Any additional packages required for the project should be listed in the requirements.txt file. Listed packages will be installed with conda. In addition, if pip install is required, add a line to the Dockerfile such as 

* RUN pip install package_name --trusted-host pypi.org --trusted-host files.pythonhosted.org

In addition, this container is loaded with the latest version of GRASS GIS. Additional GRASS GIS extensions are added by 

Before adding more, container installations (e.g., GRASSS GIS), packages with the requirements.txt file, or packages through pip install, first
* Stop running the docker container with control + c
* Execute the command 'docker-compose down' from the folder where the docker-compose file resides
* Edit the requirements in the requirements.txt file or Dockerfile
* Rebuild the container with docker-compose up

### Connecting to the Internet
If connecting to the internet with Jupyter or equivalent to download information, one first has to execute the following lines of script

* import ssl
* ssl._create_default_https_context = ssl._create_unverified_context

https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
