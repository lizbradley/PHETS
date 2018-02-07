import datetime

class randomness:
  
  suffix = 0;

  @staticmethod
  def generate_suffix():
    randomness.suffix = (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"));
  
  @staticmethod
  def get_suffix():
    
    if randomness.suffix == 0:
	randomness.generate_suffix();
    return randomness.suffix;

