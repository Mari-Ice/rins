#! /usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

from pydub import AudioSegment
from pydub.playback import play

from std_srvs.srv import Trigger

class Talker(Node):
	
	def __init__(self):
		super().__init__('talker')
	   
		pwd = os.getcwd()
		gpath = pwd[0:len(pwd.lower().split("rins")[0])+4]
		
		default_file = f"{gpath}/src/task1/sounds/obi-wan-hello-there.mp3"
		self.declare_parameter('sound_file', default_file)
		
		self.srv = self.create_service(Trigger, 'say_hello', self.say_hello_callback)
		
	def say_hello_callback(self, request, response):
		try:
			self.get_logger().info('Playing sound...')

			sound_file = self.get_parameter('sound_file').get_parameter_value().string_value
			sound = AudioSegment.from_file(sound_file)
			play(sound)
			
			response.success = True
			response.message = 'Hello there!'
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
