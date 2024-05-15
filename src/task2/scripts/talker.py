#! /usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

from pydub import AudioSegment
from pydub.playback import play
from task2.srv import Color
from std_srvs.srv import Trigger

class Talker(Node):
	
	def __init__(self):
		super().__init__('talker')
	   
	   	
		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		# reconfigure here to have all the talking in this node
		default_folder = f"{gpath}/src/task2/sounds/"


		self.color_files = [
			'red.mp3', 'green.mp3', 'blue.mp3', 'black.mp3'
		]
		self.greet = 'obi-wan-hello-there.mp3'
		self.declare_parameter('sounds_folder', default_folder)
	
		self.srv_color = self.create_service(Color, 'say_color', self.say_color_callback)
		self.srv_greet = self.create_service(Trigger, 'say_hello', self.say_hello_callback)
		
	def say_color_callback(self, request, response):
		try:
			color = request.color
			response.color = color
			response = self.play_sound(self.color_files[color], response)
		except Exception as e:
			response.success = False
			response.message = str(e)
		finally:
			return response
		
	def say_hello_callback(self, request, response):
		try:
			response = self.play_sound(self.greet, response)
		except Exception as e:
			response.success = False
			response.message = str(e)
		finally:
			return response

	def play_sound(self, sound_file, response):
		try:
			self.get_logger().info(f'Playing sound {sound_file}')

			sound_file = self.get_parameter('sounds_folder').get_parameter_value().string_value + sound_file
			sound = AudioSegment.from_file(sound_file)
			play(sound)
			
			response.success = True
			response.message = sound_file
		except Exception as e:
			response.success = False
			response.message = str(e)
		finally:
			return response

def main(args=None):
	print('Talker node starting.')

	rclpy.init(args=args)
	node = Talker()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
