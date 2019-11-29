import random

class Memory:
	def __init__(self, _memory):
		self.memory = _memory
		self.data = []

	def add_data(self, element):
		"""
			@Params: element: is an input-output sample on which the neural network will be trained
		"""
		self.data.append(element)
		length = len(self.data)
		if (length > self.memory):
			# pop the 0th element from the data list on which the neural network has already been trained on
			self.data.pop(0)


	def get_batch(self, n_elements):
		"""
			Returns: a batch of data elements for training the neural network
			@Params: n_elements: represents batch_size
		"""
		length = len(self.data)
		return random.sample(self.data, min(length, n_elements))

