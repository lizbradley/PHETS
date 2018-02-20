from misc import randomness
import os

class output_paths:
  
  @staticmethod
  def get_prfstats_samples_dir():
      suffix = randomness.get_suffix();
      try:
          path = 'output/prfstats/samples/' + suffix;
          os.makedirs(path);
      except OSError:
          pass

      return path;

  @staticmethod
  def get_prfstats_filts_path():
      
      base = 'prfstats/data/' + suffix;

      try:
          os.makedirs(base);
      except OSError:
          pass

      return (base + '/filts.npy')
      
