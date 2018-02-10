from misc import randomness

class output_paths:
  
  @staticmethod
  def get_prfstats_samples_dir():
      suffix = randomness.get_suffix();
      path = 'output/prfstats/samples/' + suffix;
      if not os.path.exists(path):
	  os.mkdir(path);
      return path;

  @staticmethod
  def get_prfstats_filts_path():
      
      base = 'prfstats/data/' + suffix;
      if not os.path.exists(base):
	  os.mkdir(base)
      return (base + '/filts.npy')
      
