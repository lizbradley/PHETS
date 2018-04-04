class GeometricInfo:
	
	global tricks
	tricks = [];
	
	@staticmethod
	def add_tricks(dog):
		tricks.append(dog)
	
	@staticmethod
	def get_tricks():
		return tricks
