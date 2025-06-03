.. LLM-GE documentation master file, created by
   sphinx-quickstart on Mon May  5 22:08:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######################################
Point Cloud Classification Optimization
#######################################

********
Overview
********

How to get started and setup your system for Point Cloud Classification Team

    In Morris, Jurado, and Zutty's recent paper "LLM Guided Evolution -- The Automation of Models Advancing Models",
    they were able to create a framework that uses LLMs with a layer of creativity to speed up the process of evolving ML models.
    Our specific goal is to generalize this framework by improving 2 current state of the start point cloud classification models.
    We picked Point Transformers and PointNet++ along with the Model40Net DataSet.

==================
Point Transformers
==================

    Point Transformers improves point cloud processing by using self-attention mechanisms, similar to those in transformer models for sequential data.
    They apply attention to capture relationships between all points in the cloud, allowing the model to focus on relevant features and dependencies, both locally and globally, leading to better handling of irregular, unordered data.
    For our purposes we're not using the EMADE Repo instead we'll be using the Large Language Model Guide Evolution Repo.

==========
PointNet++
==========

    PointNet++ extends PointNet by using a hierarchical approach to capture local geometric patterns in 3D point clouds. 
    It samples and groups points into neighborhoods, applies PointNet-style operations to each group, and then propagates the learned features to progressively capture both local and global structures.

*****
Setup
*****

=======================
Clone GitHub Repository
=======================

    Head over to Github.com and clone the Large Language Model Guided Evolution Generic Repository where we're integrating the 2 point cloud classification models.

    `LLMGE Repository Github <https://github.com/MosesTheRedSea/LLM-Guided-Evolution-Generic>`_

.. image:: point_cloud_resources/repo_screenshot.png

**Clone Repository**

    Select the main branch, when you click the green button dropdown <> Code

**Terminal Commands**

.. code-block:: console

    mkdir LLMGE_Point_Cloud

    cd LLMGE_Point_Cloud
    
    git clone https://github.com/MosesTheRedSea/LLM-Guided-Evolution-Generic.git


=======================
PACE ICE Configurations
=======================

    Training & evaluating models takes a significant amount of computing power which your computers alone may not be able to handle. Which is why this semester we will make great use of PACE ICE.

**What is PACE ICE?**

    PACE stands for Partnership for an Advanced Computing Environment, while ICE is the Instructional Cluster Environment. 

    ICE provides students and instructors with a high-performance computing environment for courses, including those in the College of Computing. 

    It's a free resource for instructional purposes. 

**Important Links**

    The Georgia Tech VPN, specifically the GlobalProtect VPN, is a secure network connection that allows users to access Georgia Tech resources and services from off-campus.

    `GATECH VPN <https://vpn.gatech.edu/global-protect/login.esp>`_

    **username** : gatech username - without @gatech.edu

    **password** : gatech account password

    .. image:: point_cloud_resources/vpn.png

    - You can access any website through the VPN through the ``enter url`` dropdown.

    .. image:: point_cloud_resources/enter_url.png

    - Enter in the pace-ice link: `<https://ondemand-ice.pace.gatech.edu/>`_

    .. note::
        You must be on the vpn to access pace.
    
    - You screen will be redirected to the pace-ice landing page.

    .. image:: point_cloud_resources/pace-ice-login.png

    - Click on the ``Flies`` dropdown in the top-left corner, then select the ``Home Directory``

    .. image:: point_cloud_resources/pace-ice-directory.png

    - Traverse all the way to your scratch directory so that we can drag and drop our LLM-GE Repository that you cloned to your local computer.
    - Click the Upload Button at the top of the window, and drag and drop your LLMGE-Generic Repository into your scratch directory.

    - Now that we finally have the repository on PACE ICE we can start the setup.

    - From here click ``Interactive Apps`` dropdown, and select ``VS Code`` the last option on the bottom.

    .. image:: point_cloud_resources/pace-ice-interactive-apps-vscode.png

    - Check the settings \(change number of hours to ``8``\) then click launch.

    .. image:: point_cloud_resources/pace-ice-vscode-job-configurations.png

    - Head back to the home page, and click on ``My Interactive Session``.

    .. image:: point_cloud_resources/pace-ice-vscode-interactive-sessions.png

    - WHen you click on the Interactive Sessions button at the top of the page, it will redirect to 
    another page that displays every current running session. 

    - The Session you just ceated has been added to the queue, one the appropiate resoruces are avaliable PACE ICE
    will let you connect and load in the session in VSCode's Web Applicaton.

    - Once you're in VSCode, simply click ``File -> Open Folder``, and go to the directory you saved the LLMGE Repo in your scratch or whichever folder.
 
    .. image:: point_cloud_resources/pace-ice-scratch-folder-directory.png

    - Before we begin integrating different architectures into LLMGE, I want to explain the codebase itself how it works, what we're trying to optimize, and our expected results.
    
=====================================
Large Language Model Guided Evolution 
=====================================

    - Large Language Model Guided Evolution is an innovative architecture that leverages the reasoning capabilities of Large Language Models to guide evolutionary search and refinement of neural network architectures. 
    
    - The system combines natural language reasoning, feedback-based learning, and genetic optimization techniques to automatically discover and improve neural architectures.

-------------------
 Codebase Breakdown
-------------------
   
    - The codebase is organized into several key directories and files, each serving specific functions in the evolutionary process:

    .. image:: point_cloud_resources/llm-ge-code-base-structure.png

    **sota**

        - sota (State of the Art) directory holds various key information we need for LLMGE to run smoothly.
        - Model Architectures: Houses the baseline neural network architectures targeted for evolutionary improvement
        - Dataset Configurations: Contains dataset specifications and preprocessing scripts
        - Training Scripts: Includes standardized training procedures and evaluation protocols

    **src**

        - The main source directory contains the core configuration files:
        - Large Language Model Configuration, Specific Directories, and Dataset paths


    **constants.py**
        
        - Centralized configuration file containing system parameters, hyperparameters, and global constants.

        .. image:: point_cloud_resources/constants-py-info-1.png

        - Seed network, data files, and train file path directories listed above for global access throughout codebase.
        
        .. image:: point_cloud_resources/constants-py-info-2.png

        - Large Language Model Configuration, along with Evolutionary Alogirthm predefined configurations, population size, number of generations, and weights.

    **templates**

        - The templates sirectory houses validation and code modification templates prompt and code generation templates used by the LLM.

        - There are a wide variety of different prompts we submit to the LLM, we have the variant templates which request for modifications to the submitted SEED file.

        - There is also a validation prompt that helps with cleaning code received from LLM, invalid syntax, unfinshed responses, invalid packages.


    **llm_crossover.py**

        .. image:: point_cloud_resources/llmge-llm-crossover-augment-network.png

        -  Implements LLM-guided crossover operations that intelligently combine features from parent architectures.

    **llm_mutation.py**

        .. image:: point_cloud_resources/llmge-llm-mutations-augment_network.png

        - Handles LLM-directed mutation operations for architecture modification and enhancement.

    **llm_utils.py**

        - Utility functions supporting LLM interactions, prompt processing, and response parsing.

        .. image:: point_cloud_resources/llmge-llm-utils-clean-code-from-llm-code.png

        - llm_utils possess all the necessary methods needed for llm code generation and quality control. 
      
        - The clean_code_from_llm has a validation sequence that ensures the quality of code recieved from the LLM, is 

        .. image:: point_cloud_resources/llmge-llm-utils-generate-code.png

        - The **generate_augmented_code** allows for a wide variety of LLM configurations, either using the hugging-face API, or a local LLM. 
        
        - You can specifiy eother by configuring the **LLM_MODAL** & **hugging_face** values in constants.py

    **run.sh**

        .. image:: point_cloud_resources/llmge-run-sh-main.png
        
        - Primary shell script for launching evolution experiments

    **run_improved.py**

        .. image:: point_cloud_resources/llmge-run-improved-code-snippet.png

        - Enhanced execution script for running evolution experiments with improved features

    **server.py**

         - Server component for distributed Local or API LLM execution

        .. image:: point_cloud_resources/llmge-local-llm-integration-server-py.png

        - The **server.py** file helps with loading the local LLM stored on PACE ICE, for us to submit the mutation prompts to.

        .. image:: point_cloud_resources/llmge-local-llm-integration-server-py-2.png

        - We load the Local LLM, so that we can submit prompts of different token lengths and get a reponse in a reasonable amount of time.
    

    **mixt.sh**

        .. image:: point_cloud_resources/llmge-mixt-sh-main.png

        - Specialized script for mixed or multi-objective evolution scenarios

--------------
Code Workflow
--------------

**Initialization**

    .. image:: point_cloud_resources/run-improved-individual-deap.png
        
    - We initialize the toolbox which holds the key methods we plan to use during the evolutionary loop.

**Evolutionary Loop**

    .. image:: point_cloud_resources/run-improved-individual-creation.png

    - This is essentially the main method within run_imrpoved that starts the model evolution. 
    - We create our population and configure key parameters to our specification. 

**Variant Model Files**

    .. image:: point_cloud_resources/create-individual-run-improved.png

    - The **create_individual method** takes the seed file which has been generically integrated and creates a varaint file with a unique geneId (20 char long random assortment of numbers & letters) in a sub-directory called llmge_models, which will be submitted fo evaluation.

    **Seed individuals**
    
    - Seed individuals are created at the very start (generation 0) using the create_individual function.


--------------------------
ExquisiteNetV2 Integration
--------------------------


**Download Dataset**

 - ExquisiteNetV2, a Lightweight CNN designed for image classification, tested on 15 datasets \(CIFAR-10, MNIST\) with 518,230 parameters, achieving 99.71% accuracy on MNIST. 

    - The default integration model for this framework is ExquisiteNetV2. We're going to go step by step file by file and show you key parts to change so please follow closely. Go into the sota  Directory and click on the ExquisiteNetV2 directory. Inside this directory you will see a README.md file which contains key information for this integration.

    **train-test data**

    .. code-block:: markdown

        # Train Cifar-10
        The best weight has been in the directory `weight/exp`.


        If you want to reproduce the result, you can follow the procedure below.
        - __Download the cifar-10 from [official website](https://www.cs.toronto.edu/~kriz/cifar.html)__
        1. Download python version and unzip it.
        2. Put `split.py` into the directory `cifar-10-python`  
            then type:
            ```
            python split.py
            ```  
            Now you get the cifar10 raw image in the directory `cifar10`
            
        - __Train from scratch__



**Constants.py**



**constants.py**



**run.sh**



**mixt.sh**



**llm_utils.py**


**llm_utils.py**


**network.py**


**train.py**



------------------------------
Point Transformers Integration
------------------------------

**Download Dataset**

**Constants.py**

**constants.py**

**run.sh**

**mixt.sh**

**llm_utils.py**

**llm_utils.py**

**network.py**

**train.py**

----------------------
PointNet++ Integration
----------------------

**Download Dataset**

**Constants.py**

**constants.py**

**run.sh**

**mixt.sh**

**llm_utils.py**

**llm_utils.py**

**network.py**

**train.py**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

