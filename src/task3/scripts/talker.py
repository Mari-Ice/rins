#! /usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

from pydub import AudioSegment
from pydub.playback import play
from task3.srv import Color, CustomMsg
from std_srvs.srv import Trigger
import speech_recognition as sr
from gtts import gTTS
import io
from enum import Enum
from task3.msg import Park
from rclpy.qos import QoSReliabilityPolicy
import time

class Talker(Node):

	def __init__(self):
		super().__init__('talker')
	   
		self.srv_color = self.create_service(Color, 'say_color', self.say_color_callback)
		self.srv_greet = self.create_service(Trigger, 'say_hello', self.say_hello_callback)
		self.srv_say_custom = self.create_service(CustomMsg, 'say_custom_msg', self.say_custom_msg_callback)
		
		self.colors = ['red', 'green', 'blue', 'black']
		self.greet = 'Hello!'
		self.ask = 'Do you know where the Mona Lisa painting is?'
		self.goodbye = 'Thank you very much. Goodbye!'

		self.srv_listen = self.create_service(Trigger, 'listen', self.listen_callback)
		self.pub_park = self.create_publisher(Park, '/park_near_obj', QoSReliabilityPolicy.BEST_EFFORT)
		self.recogniser = sr.Recognizer()

	def say_color_callback(self, request, response):
		self.get_logger().info('Playing color!')
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
			response = self.play_sound(self.ask, response)
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
			tts = gTTS(text, lang='en')
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
			# recognize speech using GoogleAPI
			while not understood:
				print("Say something!")
				audio = self.recogniser.listen(source)
				print('processing audio...')
				try:
					print("Recognizing...")
					text = self.recogniser.recognize_google(audio)
					text = text.lower()
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
		print('processing text')
		colors = []
		
		for color in self.colors:
			if color in text:
				colors.append(color)
		if len(colors) == 0:	
			response.success = False
			return 'Could not understand.'
		self.play_sound(f'I will look for it around {" and ".join(colors)} ring.', response)
		self.play_sound(self.goodbye, response)
		
		park = Park()
		park.colors = colors
		self.pub_park.publish(park)
			
		response.success = True
		return 'OK'
	def say_custom_msg_callback(self, message, response):
		try:
			response = self.play_sound(message.msg, response)
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
