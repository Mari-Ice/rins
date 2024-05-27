#! /usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

from pydub import AudioSegment
from pydub.playback import play
from task2.srv import Color
from std_srvs.srv import Trigger
import speech_recognition as sr
from gtts import gTTs
import io
from enum import Enum
from task3.msg import Park

class Talker(Node):
	
	def __init__(self):
		super().__init__('talker')
	   
		self.srv_color = self.create_service(Color, 'say_color', self.say_color_callback)
		self.srv_greet = self.create_service(Trigger, 'say_hello', self.say_hello_callback)
		
		self.colors = ['red', 'green', 'blue', 'black']
		self.greet = 'Hello!'
		self.ask = 'Do you know where the Mona Lisa painting is?'
		self.clue = 'Do you know where I should look for it?'
		self.goodbye = 'Thank you very much. Goodbye!'

		self.srv_listen = self.create_service(Trigger, 'listen', self.listen_callback)
		self.pub_park = self.create_publisher(Park, '/park_near_obj')
		self.recogniser = sr.Recognizer()

	def say_color_callback(self, request, response):
		try:
			color = self.colors[request.color]
			response.color = color
			response = self.play_sound(color, response)
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
			return response
		finally:
			response = self.listen_callback(request, response)
			return response

	def play_sound(self, text, response):
		try:
			self.get_logger().info(f'Playing sound: {text}')
			tts = gTTs(text, lang='en')
			speech_file = io.BytesIO()
			tts.write_to_fp(speech_file)
			speech_file.seek(0)
			sound = AudioSegment.from_file(speech_file, format='mp3')
			play(sound)
			
			response.success = True
			response.message = text
		except Exception as e:
			response.success = False
			response.message = str(e)
		finally:
			return response

	def listen_callback(self, request, response):
		understood = False
		with sr.Microphone() as source:
			while not understood:
				print("Say something!")
				audio = self.recogniser.listen(source)
				# recognize speech using GoogleAPI
				try:
					text = self.recogniser.recognize_google(audio)
					print("[GOOGLE]: " + text)
					understood = True
					response.message = self.process_text(text, response)
				except sr.UnknownValueError:
					print("[GOOGLE]: Could not understand audio")
					response.success = False
					response.message = 'Error'
				except sr.RequestError as e:
					print("[GOOGLE ERROR]; {0}".format(e))
					response.success = False
					response.message = e
		return response
	
	def process_text(self, text, response):
		# the function for text processing
		colors = []
		object = ''
		clue = False
		if 'No' in text or 'don\'t' in text:
			pass
		else:
			for color in self.colors:
				if color in text:
					colors.append(color)
			if 'ring' in text:
				object = 'ring'
			elif 'cylinder' in text:
				object = 'cylinder'
			else:
				response.success = False
				return 'Could not understand.'
			clue = True
		self.play_sound(self.goodbye)
		if clue:
			park = Park()
			park.color = color
			park.obj = object
			self.pub_park.publish(park)
		response.success = True
		return 'OK'

	

def main(args=None):
	print('Talker node starting.')

	rclpy.init(args=args)
	node = Talker()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
