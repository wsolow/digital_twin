# Project Title

This is the AgAid Crop Simulator for the joint [AgAid](https://agaid.org/) Project between Oregon State 
University (OSU) and Washington State University (WSU).

## Description

This package provides the following main features:
1. A crop simulation environment based off of the WOFOST8 crop simulation model
    which simulates the growth of various crops in N/P/K and water limited conditions. 
    The model has been modified to interface nicely with a Gymnasium environment
    and to provide support for perennial crops and multi-year simulations.
2. A Gymansium environment which contains the WOFOST8 crop simulation environment.
    This wrapper has support for various sized action spaces for the application
    of fertilizer and water, and various reward functions which can be specified 
    by the user. We provide easy support for multi-year simulations across various
    farms. 
3. We support the training of Deep RL agents with PPO, SAC, and DQN based on 
    [cleanRL](https://github.com/vwxyzjn/cleanrl) implementations. We provide
    various visualizations for different state features and to visualize the 
    training of RL agents.
4. The generation of historical data across years, farms and crops to support
    offline RL and off-policy evaluation methods. 

Our aim with this project is to support the AgAid community, and other researchers, 
by enabling easy evaluation of decision making systems in the agriculture environment.
Our goal is not to have the most high-fidelty simulator available, although the 
WOFOST8 model is robust, but rather to provide a realistic environment to test
RL algorithms in the agricultural setting that is easy to use and customize. 

## Getting Started

### Dependencies

* This project is entirely self contained and built to run with Python 3.10.9
* Install using miniconda3 

### Installing

Recommended Installation Method:

1. Navigate to desired installation directory
2. git clone https://github.com/wsolow/agaid_crop_simulator.git
3. conda create -n cropsim python=3.10.9
4. conda activate cropsim
5. pip install -r requirements.txt
6. pip install -e wofost_gym

These commands will install all the required packages into the conda environment
needed to run all scripts in the agaid_crop_simulator package

### Executing program

* How to run the program

After following the above installation instructions: 
1. Navigate to the base directory ../agaid_crop_simulator/
2. Run the testing domain with: python3 test_wofost.py 

When this runs successfully, to train an RL agent:
1. Run: python3 train_agent.py --agent-type <str: PPO | SAC | DQN>
2. If you have Weights and Biases set up, run:  
    python3 train_agent.py --agent-type <str: PPO | SAC | DQN> --<ag-type>.track

## Help

Initial configuration for the Gym Environment parameters (Note: NOT the actual crop simulation) 
can be modified in the utils.py file. 

This filename should show the path to the ../agaid_crop_simulator/
directory. From there, the env_config/ folder can be found with the corresponding
agromanagement, crop, and site parameters. For further information, please see the 
following READMEs:

* env_config/README_agro.md - overview of how to configure a crop simulation.

* env_config/site_config/README_add_site.md - overview of how to add a new site
    with all required parameters.
* env_config/site_config/README_site_paramters.md - an overview of all configurable site 
    parameters
* env_config/site_config/README_site_states.md - an overview of all site state and rate
    variables available for output with corresponding units.

* env_config/crop_config/README_add_crop.md - overview of how to add a new crop
    with all required parameters.
* env_config/crop_config/README_crop_paramters.md - an overview of all configurable crop 
    parameters
* env_config/crop_config/README_crop_states.md - an overview of all crop state and rate
    variables available for output with corresponding units.

* pcse/README.md - an overview of the Python Crop Simulation Environment (PCSE) and
    available resources to learn more.

* rl_algs/README.md - an overview of the available Reinforcement Learning agents
    available 

* wofost_gym/README.md - an overview of the Gymnasium wrapper and available configurations

Email soloww@oregonstate.edu with any further questions

## Authors

Will Solow (soloww@oregonstate.edu)

Dr. Sandhya Saisubramanian (sandhya.sai@oregonstate.edu)

## Version History

* 1.0.0
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

The PCSE codebase and WOFOST8 Crop Simulator can be found at:
* [pcse](https://github.com/ajwdewit/pcse)

While we made meaningful modifications to the PCSE codebase to suit our needs, 
the vast majority of the working code in the PCSE directory the property of
Dr. Allard de Wit and Wageningen-UR Group. Please see the following paper for an
overview of WOFOST:
* [wofost](https://www-sciencedirect-com.oregonstate.idm.oclc.org/science/article/pii/S0308521X17310107)

The original inspiration for a crop simulator gym environment came from the paper:
* [crop_gym](https://arxiv.org/pdf/2104.04326)

However, only a small amount of the original code was used. We have since extended
their work to interface with multiple Reinforcement Learning Agents, have added
support for perennial fruit tree crops, multi-year simulations, and different sowing
and harvesting actions. 

The Python Crop Simulation Environment (PCSE) is well documented. Resources can 
be found here:
* [pcse_docs](https://pcse.readthedocs.io/en/stable/)

The WOFOST crop simulator is also well documented, and we use the WOFOST8 model
in our crop simulator. Documentation can be found here:
* [wofost_docs](https://wofost.readthedocs.io/en/latest/)