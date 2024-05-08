#! /usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

from pydub import AudioSegment
from pydub.playback import play
from task2.srv import Color

class Talker(Node):
	
	def __init__(self):
		super().__init__('talker')
	   
	   	
		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		default_file = f"{gpath}/src/task2/sounds/"


		self.color_files = [
			'red.mp3', 'green.mp3', 'blue.mp3', 'black.mp3'
		]
		self.declare_parameter('sounds_folder', default_file)
		
		self.srv = self.create_service(Color, 'say_color', self.say_color_callback)
		
	def say_color_callback(self, request, response):
		try:
			color = request.color
			self.get_logger().info(f'Playing sound {color}')

			sound_file = self.get_parameter('sounds_folder').get_parameter_value().string_value + self.color_files[color]
			sound = AudioSegment.from_file(sound_file)
			play(sound)
			
			response.success = True
			response.message = self.color_files[color]
			response.color = color
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
