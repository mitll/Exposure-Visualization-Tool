# Exposure Visualization Tool

[![Generic badge](https://img.shields.io/badge/version-1.0-orange.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/python-3.6+-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/license-GPLv2-green)](https://shields.io/)



## Overview

The Exposure Visualization Tool is a GUI application used to view the details and results of mobile contact-tracing testing with Exposure Notification (EN) apps. The scenarios visualized involve multiple agents (phone-carriers) in a scene with EN enabled on their phone, after which a "sick" agent shares its status and an alert of exposure may be received by the other agents. This tool was created to analyze and present EN data collected by the MITLL-PACT team in collaboration with the University of Arizona. A formatted version of this data is provided in the `MITLL-UA` folder and can be viewed with this tool or used as a template for custom data. Some data files including the phone "dumpsys" logs are reduced to contain only the relevant data for this application.

### Features
- **Information panel**: Provides information on the test, "sick" and selected agents and their respective phones, including distance, phone hardware, phone placement, and exposure level. 
- **2D scene viewer**: Displays a top-down representation of the test environment and agent placements. Agents are color-coded based on the level of exposure reported by their phone, or if designated as the "sick" agent. The user can slide the simulated test time forward or backward to adjust the placement of moving agents.
- **Attenuation graph**: Shows both the average and range of signal attenuation received by the selected phone from the sick phone as a function of time, as well as attenuation threshold boundaries and the current simulated test time.

The related PACT-Beacons dataset can be found [here](https://github.com/mitll/PACT-Exposure-Notification-Beacons).

## Dependencies

Python 3.6 or greater is recommended.

Install the appropriate versions of [matplotlib](https://matplotlib.org/stable/users/installing.html), [numpy](https://numpy.org/install/), and [pygame](https://www.pygame.org/wiki/GettingStarted) for your version of python:

	$ pip install matplotlib numpy pygame

## Usage

### To run the MITLL-UA Dataset (from the root folder):

	$ python scripts/exposure_viz.py MITLL-UA/

### To run custom data:

	$ python scripts/exposure_viz.py [ROOT_DATA_PATH]
	
***Note**: For custom data, modification to variables or functions may be needed.* 

### Instructions:

- Run the Exposure Visualization tool from the command-line
- From the top bar, click on a tab to view the tests for that scenario
- Use letter keys (A-Z) to select agents by their letter ID
- Use the right arrow to move forward in time, and the left arrow to move backward in time
- Check the command-line terminal for status messages
- Scenarios using motion-capture data (such as `LargeParty`) will take longer to load

### Agent Color Key:

- Gray: no alert
- Yellow: low alert
- Red: high alert
- Green: "sick"
- Green (region): within 6 feet of "sick" phone
- Teal: default, alert type not recognized

## Data Formatting

### Directory Tree Structure:

	ROOT
	   |<scenario_01>
	   |   |<scenario_01>_setup.json
	   |   |<scenario_01>_image.png (optional)
	   |   |<test_01>
	   |   |   |<test_01>_info.json
	   |   |   |<test_01>_mocap.csv (optional)
	   |   |   |<phone_01>
	   |   |   |   |<phone_01>_dumpsys.txt
	   |   |   |<phone_02>
	   |   |   |   |<phone_02>_dumpsys.txt
	   |   |   |...
	   |   |<test_02>
	   |   |   |...
	   |   |<test_03>
	   |   |   |...
	   |<scenario_02>
	   |   |...

### "Setup" JSON (one per scenario):

- **title** : str
	- name of test to display in app
- **layout** : str
	- positioning of the test panels, either 'horizontal' or 'vertical'
- **axes** : str[2]
	- conversion of measured axes to 2D-scene axes for improved visibility
- **thresholds** : int[]
	- list of attenuation threshold values (expects at most 3)
- **mocap_type** : str (optional)
	- file extension of motion capture data, if present
- **agents**
	- agent information that is shared between tests
		- **id** : str
			- ID of the agent
		- **positions** : float[][]
			- list of 3D agent positions (feet) at each transition minute (relative to input axes)
		- **orientations** : float[]
			- list of 2D agent orientations (degrees) at each transition minute (relative to GUI axes)
		- **transition_mins** : int[]
			- minutes at which the agent changes its pose
		- **phone_situation** : str
			- carriage state of the phone
		- **phone_positions** : float[][] (optional)
			- list of 3D phone positions at each transition minute, if offset from agent
- **phones**
	- phone information that is shared between tests
		- **id** : str
			- ID of the phone
		- **model** : str
			- model of the phone
		- **calconf** : str
			- calibration confidence ("i" = unknown iPhone confidence)
		- **time_ofset** : float
			- phone time offset from the reference time, in seconds
		- **tx** : int (optional)
			- tx-power if not provided in dumpsys

### "Info" JSON (one per test):

- **testId** : str
	- ID of test used in file naming and data tables
- **startTime** : Datetime
	- Time when the test was started
- **stopTime** : Datetime
	- Time when the test was stopped
- **details**
	- test information and results
		- **agent** : str
			- ID of the agent
		- **phone** : str
			- ID of the phone held by this agent
		- **exposure** : str
			- reported exposure at the end of the test

## Citation

Please use this DOI number reference, published on Zenodo, when citing the software:

[![DOI](https://zenodo.org/badge/441285798.svg)](https://zenodo.org/badge/latestdoi/441285798)

For questions and assistance, please contact Steven Mazzola at Steven.Mazzola@ll.mit.edu

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Air Force.

© 2021 Massachusetts Institute of Technology.

	Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
	SPDX-License-Identifier: GPL-2.0-only

The software/firmware is provided to you on an As-Is basis
