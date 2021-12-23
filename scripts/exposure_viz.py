'''
© 2021 Massachusetts Institute of Technology. See LICENSE.md
Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
SPDX-License-Identifier: GPL-2.0-or-later
'''

#!/usr/bin/env python

print('')

import csv
import json
import matplotlib
import numpy as np
import os
import pygame
import sys
import time

matplotlib.use('Agg')

from datetime import datetime as dt
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.backends import backend_agg as agg



# constants
METER_TO_FOOT = 3.281 #ft/m
SEP = os.path.sep

# app colors
SCREEN_COLOR = (160, 160, 160)
TAB_ACTIVE_COLOR = (160, 160, 160)
TAB_INACTIVE_COLOR = (100, 100, 100)
TAB_BG_COLOR = (50, 50, 50)
TEST_PANEL_COLOR = (225, 225, 225)
PROXIMITY_REGION_COLOR = (200, 255, 200, 200) #rgba

# app dimensions
TAB_HEIGHT = 30
TAB_PADDING = 3
TEST_PADDING = 4
INFO_PADDING = 10
INFO_SPACING = 18
HPANEL_WIDTHS = [0.20, 0.54, 0.25]
VPANEL_HEIGHTS = [0.22, 0.48, 0.30]

# app fonts
FONTSTYLE = 'Arial'
AGENT_DESELECT_FONTSIZE = 18
AGENT_SELECT_FONTSIZE = 20
INFO_FONTSIZE = 14
TAB_FONTSIZE = 16

# 2D-scene parameters
AGENT_SIZE = 24 #px
AUTOMATIC_BOUNDS = False
PROXIMITY_RADIUS_FT = 6 #ft

# plot parameters
PLOT_LINECOLOR = 'k'
PLOT_LINEWIDTH = 1.5
PLOT_MARKERSIZE = 10
PLOT_YLIMS = (30, 90)

# misc. settings
FRAMERATE = 30 #Hz
DEFAULT_TEST_LAYOUT = 'vertical'
TIME_PAN_INTERVAL = 1 #minute



class AgentSprite(pygame.sprite.Sprite):
	""" 
	An agent sprite on the 2D-scene

	Attributes:
		base_image : pygame.Surface
			reference image for transformation
		image : pygame.Surface
			new image after transformation
		position : (int, int)
			center pixel position on 2D-scene
		label : pygame.Surface
			label to display on sprite
		label_text : str
			label text
		font_deselect : pygame.Font
			font when sprite is not selected
		font_select : pygame.Font
			font when sprite is selected	
	"""


	def __init__(self, alert, id_):
		"""
		Parameters:
			alert : str
				type of alert to associate with sprite image
			id_ : str
				text to display on sprite
		"""
		super(AgentSprite, self).__init__()
		path = os.path.dirname(sys.path[0])
		try:
			self.base_image = pygame.image.load("{}/sprites/agent_{}.png".format(path, alert)).convert()
		except FileNotFoundError:
			self.base_image = pygame.image.load("{}/sprites/agent_default.png".format(path, alert)).convert()
		self.base_image.set_colorkey((255, 255, 255), pygame.RLEACCEL)
		self.image = None
		self.position = None
		self.label_text = id_
		self.font_deselect = pygame.font.SysFont(FONTSTYLE, AGENT_DESELECT_FONTSIZE)
		self.font_select = pygame.font.SysFont(FONTSTYLE, AGENT_SELECT_FONTSIZE, bold=True)
		self.deselect()


	def setPosition(self, x, y):
		"""
		Set 2D position of sprite

		Parameters:
			x : int
				x pixel position
			y : int
				y pixel position
		"""

		self.position = (x, y)


	def select(self):
		"""Set label font to selected"""

		self.label = self.font_select.render(self.label_text, 1, (0,0,0))


	def deselect(self):
		"""Set label font to deselected"""

		self.label = self.font_deselect.render(self.label_text, 1, (0,0,0))


	def draw(self, rotation):
		"""
		Set image as the transformed base image
			
		Parameters:
			rotation : float
				degrees to rotate the sprite
		"""

		image = pygame.transform.scale(self.base_image, [AGENT_SIZE]*2)
		self.image = pygame.transform.rotate(image, rotation)



class PhoneSprite(pygame.sprite.Sprite):
	""" 
	NOT YET IMPLEMENTED
	A phone sprite on the 2D-scene
	"""

	def __init__(self):

		super(PhoneSprite, self).__init__()



class Agent:
	""" 
	Contains data for a single agent

	Attributes:
		id : str
			agent ID
		positions : float[][]
			list of 3D agent positions
		orientations : float[]
			list of sprite rotation at each position
		phase : int
			current position/orientation index
		transition_mins : float[]
			test minutes at which the phase changes
		phone_situation : str
			description of phone carriage state
		phone_positions : float[][]
			list of 3D phone positions, if available
		sprite : AgentSprite
			sprite for this agent
	"""


	def __init__(self):
		
		self.id = None
		self.positions = None
		self.orientations = None
		self.phase = 0
		self.transition_mins = None
		self.phone_situation = None
		self.phone_positions = None
		self.sprite = None



class Phone:
	""" 
	Contains data for a single phone

	Attributes:
		id : str
			phone ID
		model : str
			phone model name
		calconf : str
			phone calibration confidence
		time_offset : float
			offset in seconds of phone time to a reference time
		tx : int
			phone tx-power
		exposure : str
			alert result from exposure
		distances : float[][]
			list of distances to each sick agent
		advert_keys : str[]
			list of keys advertised by this phone
		sight_keys : str[][]
			list of keys sighted from other phones
		sight_times : float[]
			list of epoch scan times for each sighting
		sight_rssis : int[][]
			list of RSSI values collected for each sighting
		minutes : int[]
			(NOT YET IMPLEMENTED) reported exposure minutes
		sprite : PhoneSprite
			(NOT YET IMPLEMENTED) sprite for this phone
	"""


	def __init__(self):

		self.id = None
		self.model = None
		self.calconf = None
		self.time_offset = None
		self.tx = None	
		self.exposure = None
		self.distances = None
		self.sprite = None
		self.minutes = None
		self.advert_keys = []
		self.sight_keys = []
		self.sight_times = []
		self.sight_rssis = []



class Test:
	"""
	Contains data for a single test

	Attributes:
		id : str
			test ID
		path : str
			path to the test folder
		info : dict<str,>
			contains fields from the test json
		pairs : dict<Agent,Phone>
			links agents (key) to their assigned phone (value)
		t0 : datetime
			test start time
		t1 : datetime
			test end time
		duration : int
			test length in minutes
		sick_agents : int[]
			list of agents with a 'sick' phone exposure
		mocap_type : str
			file type of motion capture data, if any
		mocap : dict<string,float[][]>
			contains motion-capture position and rotation data for each agent
		mocap_t : float[]
			list of motion-capture timestamps relative to test time, in seconds
	"""	
	
	
	def __init__(self, path):
		"""
		Parameters:
			path : str
				path to the test folder
		"""

		self.path = path

		# read info JSON
		test_info_json = open(path + SEP + path.split(SEP)[-1] + '_info.json', 'r')
		self.info = json.load(test_info_json)
		self.id = self.info['testId']

		# store time values
		t0 = dt.strptime(self.info['startTime'], '%Y-%m-%dT%H:%M:%S%z')
		t1 = dt.strptime(self.info['stopTime'], '%Y-%m-%dT%H:%M:%S%z')
		self.t0 = t0
		self.t1 = t1
		self.duration = (t1-t0).seconds // 60

		# initialize unset attributes
		self.pairs = dict()
		self.sick_agents = []
		self.mocap_type = None
		self.mocap = None
		self.mocap_t = None


	def check_for_mocap(self, mocap_type):
		"""
		Read motion capture data if available in the test folder
			
		Parameters:
			mocap_type : str
				expected file type of mocap data
		"""

		mocap_path = None
		ext = mocap_type.lower()
		for filename in os.listdir(self.path):
			if filename.lower().endswith(ext):
				mocap_path = self.path + SEP + filename
				break			
		if mocap_path is not None and ext == 'csv':
			self.read_mocap_csv(mocap_path)
			self.mocap_type = ext
	

	def pair_agents_phones(self, agents, phones):
		"""
		Link agents to their held phone

		Parameters:
			agents : Agent[]
				list of agents
			phones : Phone[]
				list of phones
		"""
		
		for data in self.info['details']:
				
			# get agent and phone that match data
			agent = [agent for agent in agents if agent.id == data['agent']][0]
			phone = [phone for phone in phones if phone.id == data['phone']][0]

			# update phone fields from data
			try: # use corrected alert if available
				phone.exposure = data['realExposure']
			except KeyError:
				phone.exposure = data['exposure']
			try:
				phone.minutes = data['minutes']
			except KeyError: pass
			
			if phone.exposure == 'sick':
				self.sick_agents.append(agent)

			self.pairs[agent] = phone


	def compute_distances(self):
		"""Compute dynamic distances from sick phones to other phones"""

		# get length of distance list
		if self.mocap_type == 'csv':
			mocap_min = [i / 60.0 for i in self.mocap_t] # convert to minutes
			mocap_min_in_range = [i for i in mocap_min if i >= 0 and i <= self.duration]
			n_dists = len(mocap_min_in_range)
		else:
			n_dists = self.duration + 1

		for agent1, phone1 in self.pairs.items():
			phone1.distances = []
			for agent2 in self.sick_agents:
				if agent1 == agent2:
					dists = [0] * n_dists
				else:
					dists = [None] * n_dists
					i1, i2 = [0, 0] # initialize position indices
					for j in range(n_dists):
						# get next timestamp
						if self.mocap_type == 'csv': 
							t = mocap_min_in_range[j]
						else:
							t = j
						# get phone positions at this timestamp
						for k in range(i1, len(agent1.transition_mins)-1):
							if agent1.transition_mins[k+1] > t:
								i1 = k
								break
							elif k == len(agent1.transition_mins)-2:
								i1 = k+1
						for k in range(i2, len(agent2.transition_mins)-1):
							if agent2.transition_mins[k+1] > t:
								i2 = k
								break
							elif k == len(agent2.transition_mins)-2:
								i2 = k+1
						if agent1.phone_positions is not None:
							p1 = agent1.phone_positions[i1]
						else:
							p1 = agent1.positions[i1]
						if agent2.phone_positions is not None:
							p2 = agent2.phone_positions[i2]
						else:
							p2 = agent2.positions[i2]
						if p1 == [] or p2 == []: continue
						# compute distance
						d = ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )**0.5
						d = round(d, 1)
						dists[j] = d
				# store list of distances to this sick phone
				phone1.distances.append(dists)


	def read_dumpsys(self):
		"""Read and store packet data from phone dumpsys files"""

		# gather all dumpsys paths
		dumpsys_files = []
		for _, dirs, _, in os.walk(self.path):
			for phone_dir in sorted(dirs):
				phone_path = self.path + SEP + phone_dir
				for _, _, files in os.walk(phone_path):
					for filename in files:
						if 'dumpsys' in filename:
							dumpsys_path = phone_path + SEP + filename
							dumpsys_files.append(dumpsys_path)
							break
					break
			break
		
		phones = [phone for phone in self.pairs.values()]
		for i, dumpsys_path in enumerate(dumpsys_files):
			dumpsys_name = dumpsys_path.split('/')[-1]
			try: # get phone that matches this file
				p = [j for j, phone in enumerate(phones) if dumpsys_name.startswith(phone.id)][0]
			except IndexError:
				print('\tNo phone match for \'{}\''.format(dumpsys_name))
				continue
			phone = phones[p]
			# parse dumpsys file
			adverts = False
			sights = False
			metadata = ''
			txt_file = open(dumpsys_path, 'r')
			for line in txt_file:
				line = line.strip()
				if line == 'Advertisement records:':
					adverts = True
				elif line == 'Sighting records:':
					adverts = False
					sights = True
				elif line.startswith('tx_calibration_power'):
					phone.tx = int(line.split(' ')[2])
				elif adverts:
					phone.advert_keys.append(line.split('=')[-1])
				elif sights:
					if line.startswith('RPI'):
						sight_key = line.split('=')[-1]
					else:
						try:
							vals = line.split()
							try:
								stamp_dt = dt.strptime(vals[0][:-3], '%Y-%m-%dT%H:%M:%S')
							except ValueError:
								stamp_dt = dt.strptime(vals[0][:-5], '%Y-%m-%dT%H:%M:%S')
							stamp = time.mktime(stamp_dt.timetuple()) + 0.5
							rssi_list = ''.join(vals[2:])[6:-1].split(',')
							rssi_list = [int(r) for r in rssi_list]
							try:
								duration = int(vals[1][1:-2])
								lin = np.linspace(0, duration, len(rssi_list))
								times = [stamp + phone.time_offset + l for l in lin]
							except ValueError: pass
							phone.sight_keys.append(sight_key)
							phone.sight_times.append(times)
							phone.sight_rssis.append(rssi_list)
						except IndexError: continue

		# check for iPhones with unknown advert keys
		p = [j for j, phone in enumerate(phones) if phone.calconf == 'i']
		if len(p) == 1:
			# if only 1 iPhone in test, assume unassigned keys
			i = p[0]
			iphone = phones[i]
			iphone_keys = []
			all_keys = []
			for j, phone in enumerate(phones):
				for key in phone.sight_keys:
					if key not in all_keys:
						all_keys.append(key)
			for j, phone in enumerate(phones):
				if j == i: continue
				all_keys = [key for key in all_keys if key not in phone.advert_keys]
			iphone.advert_keys = all_keys
		elif len(p) > 1:
			print('\tMultiple iPhones found, cannot assume keys')


	def read_mocap_csv(self, mocap_path):
		"""
		Read and store motion-capture data from csv file

		Parameters:
			mocap_path : str
				path to motion-capture data
		"""

		mocap = dict()
		mocap_t = []

		with open(mocap_path) as csv_file:
			reader = csv.reader(csv_file, delimiter=',')
			for r, row in enumerate(reader):

				if r == 14:
					# get mocap start time and offset from test start time
					t0_dt = dt.strptime(row[11][:26], '%Y-%m-%d %I.%M.%S.%f %p')
					t0 = time.mktime(t0_dt.timetuple()) + t0_dt.microsecond/1E6
					t0_viz = time.mktime(self.t0.timetuple())
					delta = t0 - t0_viz

				elif r == 17:
					# get headers
					headers = []
					for h in row[2:]:
						if h not in headers:
							headers.append(h)
							mocap[h] = []
					has_data = [False]*len(headers)
					
				elif r >= 21:
					# record timestamp
					t = float(row[1]) + delta
					mocap_t.append(t)
					# record agent vectors
					for c in range(2, 2+len(headers)*7, 7):
						i = (c-2)//7
						try:
							pos = [float(p)*METER_TO_FOOT for p in row[c+4:c+7]]
							pos = [pos[0], pos[2], pos[1]]
							rot = [float(q) for q in row[c:c+4]]
							rot = [rot[0], rot[2], rot[1], rot[3]]
							if None in pos or None in rot:
								raise ValueError			
						except ValueError:
							if has_data[i]:
								# use data from previous row
								pos, rot = mocap[headers[i]][-1]
							else: continue
						if not has_data[i]:
							# set previous rows to current row
							for j in range(r-21):
								mocap[headers[i]].append([pos, rot])
							has_data[i] = True
						mocap[headers[i]].append([pos, rot])

		self.mocap = mocap
		self.mocap_t = mocap_t


	def get_agent(self, id_):
		"""
		Return agent with the given ID

		Parameters:
			id_ : str
				requested agent ID
		Return:
			agent : Agent
				agent that matches ID, or None if no match found
		"""
		
		for agent in self.pairs.keys():
			if agent.id == id_:
				return agent
		return None



class TestPanel:
	"""
	Contains graphics for displaying a single test

	Attributes:
		base : pygame.Surface
			base panel where graphical items are contained
		label : pygame.Label
			label for test information text
		font : pygame.Font
			label font
		scene : pygame.Surface
			surface for 2D-scene and agent sprites
		xMin : int
			minimum 3D x-position to show
		xMax : int
			maximum 3D x-position to show
		yMin : int
			minimum 3D y-position to show
		yMax : int
			maximum 3D y-position to show
		ratio : float
			number of pixels per foot
		neg_x : bool
			whether to negate 3D x-axis
		neg_y : bool
			whether to negate 3D y-axis
		flip_xy : bool
			whether to flip 3D x and y axes
		selected : Agent
			the currently selected agent
		fig : Figure
			figure containing the attenuation plot axes
		plot_image : pygame.Surface
			image of the figure
		plot_lines : Line2D[]
			list of plotted attenuation lines in the figure
		plot_timeline : Line2D
			time indicator line in the figure
		base_position : int[2]
			pixel position of the base panel
		label_positions : int[2][2]
			pixel positions of the information labels
		scene_position : int[2]
			pixel position of the 2D-scene
		plot_position : int[2]
			pixel position of the plot image	
	"""	

	def __init__(self):

		self.font = pygame.font.SysFont(FONTSTYLE, INFO_FONTSIZE)
		self.base = None
		self.label = None
		self.scene = None
		self.selected = None
		self.fig = None
		self.plot_image = None
		self.plot_lines = None
		self.plot_timeline = None
		

	def draw_base(self, width, height):
		"""
		Create the base panel

		Parameters:
			width : int
				width of base in pixels
			height : int
				height of base in pixels
		"""

		self.base = pygame.Surface((width, height))
		self.base.fill(TEST_PANEL_COLOR)


	def init_scene(self, pairs, axes_def, test_id=None):
		"""
		Create sprites, set axes limits and directions

		Parameters:
			pairs : dict<Agent,Phone>
				agent/phone pairs
			axes_def : str
				relation between test and scene axes
			test_id : str
				ID of test
		"""

		self.selected = None
		self.neg_x = False
		self.neg_y = False
		self.flip_xy = False

		for agent, phone in pairs.items():
			# create sprites
			agent.sprite = AgentSprite(phone.exposure, agent.id)
			phone.sprite = PhoneSprite()

		if AUTOMATIC_BOUNDS or test_id == None:
			# find position extremes
			xMin = np.inf
			xMax = -np.inf
			yMin = np.inf
			yMax = -np.inf
			for agent, phone in pairs.items():
				this_xMin = min([pos[0] if pos != [] else np.inf for pos in agent.positions])
				this_xMax = max([pos[0] if pos != [] else -np.inf for pos in agent.positions])
				this_yMin = min([pos[1] if pos != [] else np.inf for pos in agent.positions])
				this_yMax = max([pos[1] if pos != [] else -np.inf for pos in agent.positions])
				if this_xMin < xMin: xMin = this_xMin
				if this_xMax > xMax: xMax = this_xMax
				if this_yMin < yMin: yMin = this_yMin
				if this_yMax > yMax: yMax = this_yMax
		else:
			# manually set limits for each test scenario
			if test_id.startswith('als'):
				xMin, xMax, yMin, yMax = (0, 18, 0, 29)
			elif test_id.startswith('aud'):
				xMin, xMax, yMin, yMax = (-1, 70, -1, 71)
			elif test_id.startswith('bus_q'):
				xMin, xMax, yMin, yMax = (-2, 22, 0, 0)
			elif test_id.startswith('large_party'):
				xMin, xMax, yMin, yMax = (-3, 29, -4, 20)
			elif test_id.startswith('mbta_bus'):
				xMin, xMax, yMin, yMax = (-3, 8, -1, 56)
			elif test_id.startswith('train'):
				xMin, xMax, yMin, yMax = (-1, 9, -1, 64)
			elif test_id.startswith('shuttle'):
				xMin, xMax, yMin, yMax = (-6, 29, -1, 9)
			elif test_id.startswith('small_hall'):
				xMin, xMax, yMin, yMax = (-2, 34, 0, 45)
			elif test_id.startswith('small_party'):
				xMin, xMax, yMin, yMax = (0, 15, 0, 19)

		xMin = np.floor(xMin)
		xMax = np.ceil(xMax)
		yMin = np.floor(yMin)
		yMax = np.ceil(yMax)

		# add padding for 1D tests
		if xMin == xMax:
			xMin -= 1
			xMax += 1
		if yMin == yMax:
			yMin -= 1
			yMax += 1

		if '-x' in axes_def:
			tmp = -xMin
			xMin = -xMax
			xMax = tmp
			self.neg_x = True
		if '-y' in axes_def:
			tmp = -yMin
			yMin = -yMax
			yMax = tmp
			self.neg_y = True
		if axes_def[0].endswith('y') and axes_def[1].endswith('x'):
			tmp = xMin
			xMin = yMin
			yMin = tmp
			tmp = xMax
			xMax = yMax
			yMax = tmp
			self.flip_xy = True

		self.xMin = xMin
		self.xMax = xMax
		self.yMin = yMin
		self.yMax = yMax
		

	def draw_scene(self, maxWidth, maxHeight, image=None):
		"""
		Create the 2D-scene surface using axes limits or a provided image
		
		Parameters:
			maxWidth : float
				maximum allowed width of the scene in pixels
			maxHeight : float
				maximum allowed height of the scene in pixels
			image :	pygame.Surface
				background image of scene
		"""

		if image != None:
			imageWidth = image.get_width()
			imageHeight = image.get_height()
			factor = min(maxWidth/imageWidth, maxHeight/imageHeight) # preserve constraints and aspect ratio
			sceneWidth = int(imageWidth * factor)
			sceneHeight = int(imageHeight * factor)
			self.scene = pygame.transform.scale(image, (sceneWidth, sceneHeight))
		else:
			imageWidth = self.xMax - self.xMin
			imageHeight = self.yMax - self.yMin
			factor = min(maxWidth/imageWidth, maxHeight/imageHeight) # preserve constraints and aspect ratio
			sceneWidth = int(imageWidth * factor)
			sceneHeight = int(imageHeight * factor)
			self.scene = pygame.Surface((sceneWidth, sceneHeight))
			self.scene.fill((255,255,255)) # blank background if no image provided

		# compute pixel-to-distance ratio
		fx = sceneWidth / (self.xMax - self.xMin)
		fy = sceneHeight / (self.yMax - self.yMin)
		self.ratio = min(fx, fy)

		
	def draw_agents(self, pairs):
		"""
		Set position and rotation of agent sprites
		
		Parameters:
			pairs : dict<Agent,Phone>
				agent/phone pairs
		"""

		for agent, phone in pairs.items():
			if agent.positions[agent.phase] == []:
				# place sprite outside of 2D-scene
				x = np.inf
				y = np.inf
			else:
				ax = agent.positions[agent.phase][0]
				ay = agent.positions[agent.phase][1]
				if self.neg_x:
					ax = -ax
				if self.neg_y:
					ay = -ay
				if self.flip_xy:
					tmp = ax
					ax = ay
					ay = tmp
				# place sprite on 2D-scene
				x = self.ratio * (ax - self.xMin)
				y = self.scene.get_height() - self.ratio * (ay - self.yMin)
			agent.sprite.setPosition(x,y)
			agent.sprite.draw(agent.orientations[agent.phase] - 90)

	
	def init_plot(self, tMax, thresholds):
		"""
		Create figure for plot and initialize axes 

		Parameters:
			tMax : int
				maximum time value of the horizontal axis
			thresholds : int[3]
				attenuation thresholds where plot color changes
		"""

		self.fig = plt.figure(figsize=[0.01, 0.01], dpi=100)
		self.plot_image = None
		self.plot_lines = []
		self.plot_timeline = None

		ax = self.fig.add_subplot(111)
		ax.set_xlim(0, tMax+1)
		ax.set_ylim(PLOT_YLIMS)
		ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
		ax.set_xlabel('Time from Start (min)')
		ax.set_ylabel('Exposure Atten. (dBm)')
		try:
			ax.axhspan(PLOT_YLIMS[0], thresholds[0], color='red', alpha=0.1)
			ax.axhspan(thresholds[0], thresholds[1], color='yellow', alpha=0.1)
			ax.axhspan(thresholds[1], thresholds[2], color='green', alpha=0.1)
			ax.axhspan(thresholds[2], PLOT_YLIMS[1], color='blue', alpha=0.1)
		except IndexError: pass
		ax.grid()
		canvas = agg.FigureCanvasAgg(self.fig)
		canvas.draw()


	def draw_plot(self, width, height):
		"""
		Resize figure and create plot image

		Parameters:
			width : float
				figure width in pixels
			height : float
				figure height in pixels
		"""

		self.fig.set_size_inches(width*0.01, height*0.01)
		self.fig.tight_layout()
		canvas = agg.FigureCanvasAgg(self.fig)
		canvas.draw()
		renderer = canvas.get_renderer()
		raw_data = renderer.tostring_rgb()
		size = canvas.get_width_height()
		self.plot_image = pygame.image.fromstring(raw_data, size, 'RGB')


	def plot_atten(self, phone_a, phone_s, t0):
		"""
		Compute and plot the attenuation between two phones

		Parameters:
			phone_a : Phone
				receiver phone
			phone_s : Phone
				sender phone
			t0 : datetime
				start time of the horizontal axis
		"""
		
		# clear old lines
		for line in self.plot_lines:
			line.remove()
		self.plot_lines = []

		# get packets from phone_a that were sent by phone_s
		i_sights = [i for i, key in enumerate(phone_a.sight_keys) if key in phone_s.advert_keys]
		sight_times = [phone_a.sight_times[i] for i in i_sights]
		sight_rssis = [phone_a.sight_rssis[i] for i in i_sights]
		if sight_times == []: return

		label = '{}->{}'.format(phone_a.id, phone_s.id)
		times = []
		attens = []
		emins = []
		emaxs = []
		t0_int = int(time.mktime(t0.timetuple()))
		for stamps in sight_times:
			if stamps == []: continue
			times.append((stamps[0] - t0_int)/60.0) # convert to minutes
		for burst in sight_rssis:
			if burst == []: continue
			atten_burst = [phone_s.tx - b for b in burst]
			# store average attenuation
			atten_avg = np.average(atten_burst)
			attens.append(atten_avg)
			# store range of attenuation
			emins.append(atten_avg - min(atten_burst))
			emaxs.append(max(atten_burst) - atten_avg)
		
		m = self.fig.gca().plot(times, attens, '.', color=PLOT_LINECOLOR, markersize=PLOT_MARKERSIZE, zorder=10, label=label)[0]
		e = self.fig.gca().errorbar(times, attens, [emins,emaxs], linewidth=PLOT_LINEWIDTH, color=PLOT_LINECOLOR, ecolor=PLOT_LINECOLOR, zorder=10, label='_nolegend_')
		self.plot_lines = [m,e]


	def write_info(self, test, minute):
		"""
		Set the information label text

		Parameters:
			test : Test
				the test to pull information from
			minute : int
				the current test minute
		"""

		text = [[],[]] # two columns
		text[0].append('Test: ' + test.id)
		text[0].append('No. Agents: ' + str(len(test.pairs.keys())))
		for i, sick_agent in enumerate(test.sick_agents):
			sick_phone = test.pairs[sick_agent]
			j = ' ' + str(i+1) if len(test.sick_agents) > 1 else ''
			text[0].append('Sick' +  j  + ':')
			text[0].append('  Agent: ' + sick_agent.id)
			text[0].append('  Phone: ' + sick_phone.id)
			text[0].append('  Model: ' + sick_phone.model)
			text[0].append('  CalConf: ' + sick_phone.calconf)
			text[0].append('  Carriage: ' + sick_agent.phone_situation)
		if self.selected == None:
			text[1].append('Selected Agent: None')
		else:
			selected_agent = self.selected
			selected_phone = test.pairs[self.selected]
			text[1].append('Selected:')
			for i, sick_agent in enumerate(test.sick_agents):
				j = ' ' + str(i+1) if len(test.sick_agents) > 1 else ''
				text[1].append('  Distance' + j + ': ' + str(selected_phone.distances[i][minute]) + ' ft')
			text[1].append('  Exposure: ' + selected_phone.exposure.upper())
			text[1].append('  Agent: ' + selected_agent.id)
			text[1].append('  Phone: ' + selected_phone.id)
			text[1].append('  Model: ' + selected_phone.model)
			text[1].append('  CalConf: ' + selected_phone.calconf)
			text[1].append('  Carriage: ' + selected_agent.phone_situation)
			
		self.label = [[],[]]
		for t in text[0]:
			self.label[0].append(self.font.render(t, 1, (0,0,0)))
		for t in text[1]:
			self.label[1].append(self.font.render(t, 1, (0,0,0)))


	def select(self, agent):
		"""
		Set the given agent as the selected agent

		Parameters:
			agent : Agent
				the agent to be selected		
		"""

		if self.selected != None:
			self.selected.sprite.deselect()
		agent.sprite.select()
		self.selected = agent



class Scenario:
	"""
	Contains data relevant to all tests of a scenario

	Attributes:
		path : str
			path to the scenario folder
		setup : dict<str,>
			contains fields from the setup json
		name : str
			name of the scenario
		layout : str
			defines how test panels are displayed, either 'horizontal' or 'vertical'
		tests : Test[]
			list of tests
		panels : TestPanel[]
			list of test panels
		test_widths : float[]
			list of test panel widths
		test_heights : float[]
			list of test panel heights
		image : pygame.Surface
			background image for the 2D-scene of all tests
		test_minute : int
			current displayed time in minutes of all tests
		max_test_minute : int
			maximum time in minutes of all tests
	"""

	def __init__(self, path):
		"""
		Parameters:
			path : str
				path to the scenario folder
		"""

		self.path = path
		self.setup = None
		self.name = None
		self.layout = None
		self.tests = []
		self.panels = []
		self.test_widths = None
		self.test_heights = None
		self.image = None
		self.test_minute = 0
		self.max_test_minute = 0
		
		try: # load setup file
			filename = [f for f in os.listdir(path) if f.endswith('setup.json')][0]
			_file = open(path + SEP + filename, 'r')
			self.setup = json.load(_file)
		except IndexError:
			print('\t{}: missing setup file.'.format(name))
			return
		if 'title' in self.setup.keys():
			self.name = self.setup['title']
		else:
			self.name = path.split(SEP)[-1]
		if 'layout' in self.setup.keys():
			self.layout = self.setup['layout']
		else:
			self.layout = DEFAULT_TEST_LAYOUT
		
		try: # load scene image
			filename = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))][0]
			self.image = pygame.image.load(path + SEP + filename).convert()
		except IndexError:
			print('\t{}: no image file.'.format(self.name))

	
	def load_tests(self):
		"""Load all test data and initialize test containers"""

		self.tests = []
		print('Loading {}:'.format(self.name))
		for _, dirs, _, in os.walk(self.path):
			for d in sorted(dirs):
				print('{}...'.format(d))
				print('\tLoading test data...')
				test = Test(self.path + SEP + d)
				if 'mocap_type' in self.setup.keys():
					test.check_for_mocap(self.setup['mocap_type'])
				if test.mocap is not None:
					agents = self.init_agents(test.mocap, test.mocap_t)
				else:
					agents = self.init_agents()
				phones = self.init_phones()
				print('\tPairing agents to phones...')
				test.pair_agents_phones(agents, phones)
				print('\tComputing phone distances...')
				test.compute_distances()
				print('\tReading RSSI data...')
				test.read_dumpsys()
				print('\tDone.')
				self.tests.append(test)
			break
		self.max_test_minute = max([test.duration for test in self.tests])


	def init_agents(self, mocap=None, mocap_t=None):
		"""
		Initialize agents from setup file
		
		Parameters:
			mocap : dict<str,float[][]>
				contains motion capture position and rotation data for each agent
			mocap_t : float[]
				list of motion capture timestamps relative to test time, in seconds
		Return:
			agents : Agent[]
				list of initialized agents
		"""
		
		agents = []
		for data in self.setup['agents']:
			agent = Agent()
			agent.id = data['id']
			agent.phone_situation = data['phone_situation']
			if mocap is not None:
				key = agent.id.lower() + '_agent'
				agent.positions = [v[0] for v in mocap[key]]
				agent.orientations = []
				qs = [v[1] for v in mocap[key]]
				for x, y, z, w in qs:
					t3 = 2*(w*z + x*y)
					t4 = 1 - 2*(y*y + z*z)
					yaw = -np.arctan2(t4, t3)
					agent.orientations.append(yaw * 180 / np.pi + 90)
				agent.transition_mins = [t / 60.0 for t in mocap_t] # convert to minutes				
				agent.phone_positions = None
			else:	
				agent.positions = data['positions']
				agent.orientations = data['orientations']
				agent.transition_mins = data['transition_mins']
				try:
					agent.phone_positions = data['phone_positions']
				except KeyError:
					agent.phone_positions = None
			agents.append(agent)

		return agents


	def init_phones(self):
		"""
		Initialize phones from setup file

		Return:
			phones : Phone[]
				list of initialized phones		
		"""		
		
		phones = []	
		for data in self.setup['phones']:
			phone = Phone()
			phone.id = data['id']
			phone.model = data['model']
			phone.calconf = data['calconf']
			phone.time_offset = data['time_offset']
			try:
				phone.tx = data['tx']
			except KeyError:
				phone.tx = None
			phones.append(phone)

		return phones


	def init_panels(self):
		"""Create test panels and initialize parameters"""

		self.panels = []
		self.test_minute = 0
		
		n = float(len(self.tests))
		if n == 0: return

		for test in self.tests:
			panel = TestPanel()
			panel.init_scene(test.pairs, self.setup['axes'], test.id)
			panel.init_plot(self.max_test_minute, self.setup['thresholds'])	
			self.panels.append(panel)

		if self.layout == 'horizontal':
			x1, x2, x3 = HPANEL_WIDTHS
			self.test_widths = lambda w : [x1*w, x2*w, x3*w]
			self.test_heights = lambda h : [(h - TAB_HEIGHT)/n - TEST_PADDING]*3
		else:
			y1, y2, y3 = VPANEL_HEIGHTS
			self.test_widths = lambda w : [w/n - TEST_PADDING]*3
			self.test_heights = lambda h :  [y1*(h - TAB_HEIGHT), y2*(h - TAB_HEIGHT), y3*(h - TAB_HEIGHT)]
		
		self.change_phase()


	def update(self, sw, sh):
		"""
		Update test panels

		Parameters:
			sw : int
				current width of the app window
			sh : int
				current height of the app window
		"""

		widths = self.test_widths(sw)
		heights = self.test_heights(sh)

		for i, panel in enumerate(self.panels):
			panel.write_info(self.tests[i], min(self.tests[i].duration, self.test_minute))
			panel.draw_scene(widths[1], heights[1], self.image)
			panel.draw_agents(self.tests[i].pairs)
			panel.draw_plot(widths[2], heights[2])
			panel.label_positions = [(INFO_PADDING, INFO_PADDING + j*INFO_SPACING) for j in range(len(panel.label[0]))] \
								 + [(INFO_SPACING + widths[0]/2, INFO_PADDING + j*INFO_SPACING) for j in range(len(panel.label[1]))]
			if self.layout == 'horizontal':
				panel.base_position = (0, i*((sh-TAB_HEIGHT)/3.0 + TEST_PADDING) + TAB_HEIGHT)
				panel.scene_position = (widths[0] + 0.5*(widths[1]-panel.scene.get_width()), 0.5*(heights[1] - panel.scene.get_height()))
				panel.plot_position = (sw - widths[2], 0)
				panel.draw_base(sw, max(heights))
			else:
				panel.base_position = (i*(sw/3.0 + TEST_PADDING), TAB_HEIGHT)
				panel.scene_position = (0.5*(widths[1]-panel.scene.get_width()), heights[0] + 0.5*(heights[1] - panel.scene.get_height()))
				panel.plot_position = (0, sh - heights[2])
				panel.draw_base(max(widths), sh)
			
			
	def release_panels(self):
		"""Delete test panels to free up memory"""

		self.test_minute = 0
		self.change_phase()
		for panel in self.panels:
			plt.close(panel.fig)
		self.panels = []


	def change_phase(self):
		"""Set agent phase and move plot timeline using the current test minute"""

		for i, test in enumerate(self.tests):
			panel = self.panels[i]
			for agent in test.pairs.keys():
				agent.phase = [j for j, m in enumerate(agent.transition_mins) if m <= self.test_minute][-1]
			if panel.plot_timeline != None:
				panel.plot_timeline.remove()
			panel.plot_timeline = panel.fig.gca().axvline(x=self.test_minute, color='r')


class Tab():
	"""
	Container class for a scenario tab button
	
	Parameters:
		surf : pygame.Surface
			base surface
		label : pygame.Label
			label containing the display text
		text : str
			label text
		font : pygame.Font
			label font
		index : int
			index of this tab in a tab group
		position : int[2]
			pixel position of surface
		label_position : int[2]
			pixel position of label
	"""

	def __init__(self, text, width, index):
		"""
		Parameters:
			text : str
				text to display
			width : int
				tab width
			index : int
				tab group index
		"""		

		self.font = pygame.font.SysFont(FONTSTYLE, TAB_FONTSIZE)
		self.text = text
		self.label = self.font.render(text, 1, (0,0,0))
		self.index = index
		self.update(width)


	def update(self, width):
		"""
		Scale tab to fit in width

		Parameters:
			width : int
				new width of the surface
		"""		

		self.surf = pygame.Surface((width, TAB_HEIGHT))
		self.position = (self.index*(width + TAB_PADDING), 0)
		self.label_position = ((width - self.label.get_width()) // 2 , (TAB_HEIGHT - self.label.get_height()) // 2)



# === Main Loop ===

print('\nStarting Exposure Visualization...')

try: # get root data path from first arg
	path = sys.argv[1]
except IndexError:
	print('Please specify data path as argument.')
	exit()

if path[-1] != SEP:
	path += SEP
		
# initialize pygame
pygame.init()
pygame.display.set_caption('Exposure Visualization')
screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)
screen_width, screen_height = screen.get_rect().size
clock = pygame.time.Clock()

# initialize scenarios
scenario_list = []
for _, dirs, _, in os.walk(path):
	for d in sorted(dirs):
		scenario_list.append(Scenario(path + d))
	break

# link scenarios to tabs
tab_width = lambda: max(1, screen_width/len(scenario_list) - TAB_PADDING*(1 - 1/len(scenario_list)))
tab_bar = pygame.Surface((screen_width, TAB_HEIGHT))
scenarios = []
for i, scenario in enumerate(scenario_list):
	tab = Tab(scenario.name, tab_width(), i)
	scenarios.append((tab, scenario))
active = None

# pygame loop
running = True
while running:
	delta = clock.tick(FRAMERATE)/1000.0
	
	for event in pygame.event.get():

		if event.type == pygame.QUIT:
			running = False
		
		elif event.type == pygame.VIDEORESIZE:
			screen_width, screen_height = screen.get_rect().size
			for tab, _ in scenarios:
				tab.update(tab_width())
		
		elif event.type == pygame.MOUSEBUTTONDOWN:
			mouse_x, mouse_y = pygame.mouse.get_pos()
			if mouse_y <= TAB_HEIGHT:
				for i, (tab, scenario) in enumerate(scenarios):
					# check if tab was clicked
					x0 = tab.position[0]
					if mouse_x >= x0 and mouse_x <= x0 + tab_width():
						# load tests if needed
						if scenario.tests == []:
							scenario.load_tests()
						# release last tab's items from memory
						if active is not None:
							scenarios[active][1].release_panels()
						active = i
						# initialize selected tab's items
						scenario.init_panels()
						break
		
		elif event.type == pygame.KEYDOWN:
		
			if event.key == pygame.K_ESCAPE:
				running = False

			elif event.key == pygame.K_LEFT:
				# move test time backward
				scenarios[active][1].test_minute = max(0, scenarios[active][1].test_minute - TIME_PAN_INTERVAL)
				scenarios[active][1].change_phase()

			elif event.key == pygame.K_RIGHT:
				# move test time forward
				scenarios[active][1].test_minute = min(scenarios[active][1].test_minute + TIME_PAN_INTERVAL, scenarios[active][1].max_test_minute)
				scenarios[active][1].change_phase()

			else:
				key_char = pygame.key.name(event.key).upper()
				if active is not None:
					scenario = scenarios[active][1]
					for i, test in enumerate(scenario.tests):
						panel = scenario.panels[i]
						# get agent based on key pressed
						agent = test.get_agent(key_char)
						if agent is not None:
							panel.select(agent)
							phone_a = test.pairs[agent]
							phone_s = test.pairs[test.sick_agents[0]]  # assuming 1 sick agent
							panel.plot_atten(phone_a, phone_s, test.t0)

	# update screen
	screen.fill(SCREEN_COLOR)

	# update tab bar
	tab_bar.fill(TAB_BG_COLOR)
	for i, (tab, _) in enumerate(scenarios):
		tab.surf.fill(TAB_ACTIVE_COLOR if i == active else TAB_INACTIVE_COLOR)	
		tab.surf.blit(tab.label, tab.label_position)
		tab_bar.blit(tab.surf, tab.position)
	screen.blit(tab_bar, (0, 0))
	
	if active is not None:
		# update active scenario
		scenario = scenarios[active][1]
		scenario.update(screen_width, screen_height)
		for i, panel in enumerate(scenario.panels):
			test = scenario.tests[i]
			# blit test labels
			for j, label in enumerate(panel.label[0] + panel.label[1]):
				panel.base.blit(label, panel.label_positions[j])
			# blit sick agent proximity regions
			for sick_agent in test.sick_agents:
				radiusSurf = pygame.Surface((2*PROXIMITY_RADIUS_FT*panel.ratio, 2*PROXIMITY_RADIUS_FT*panel.ratio), pygame.SRCALPHA)
				pygame.draw.circle(radiusSurf, PROXIMITY_REGION_COLOR, (PROXIMITY_RADIUS_FT*panel.ratio,PROXIMITY_RADIUS_FT*panel.ratio), PROXIMITY_RADIUS_FT*panel.ratio)
				panel.scene.blit(radiusSurf, (sick_agent.sprite.position - PROXIMITY_RADIUS_FT*panel.ratio))
			# blit agent sprites to scene view
			for agent, phone in test.pairs.items():
				panel.scene.blit(agent.sprite.image, agent.sprite.image.get_rect(center=agent.sprite.position))
				panel.scene.blit(agent.sprite.label, agent.sprite.label.get_rect(center=agent.sprite.position))
				if panel.selected != None:
					# blit distance lines from selected agent to sick agents
					for sick_agent in test.sick_agents:
						pygame.draw.line(panel.scene, (0,0,0), panel.selected.sprite.position, sick_agent.sprite.position)
			# blit child panels to parent
			panel.base.blit(panel.scene, panel.scene_position)
			panel.base.blit(panel.plot_image, panel.plot_position)
			screen.blit(panel.base, panel.base_position)
	
	pygame.display.flip() # refresh

pygame.quit()
