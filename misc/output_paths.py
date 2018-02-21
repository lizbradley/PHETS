from misc import randomness
import os
from utilities import make_dir

class output_paths:
  
  @staticmethod
  def get_prfstats_path(out_filename):
      
      out_dir, out_filename = os.path.split(out_filename)
      suffix = randomness.get_suffix();
      out_dir = out_dir + '/' + suffix
      make_dir(out_dir)
      out_filename = out_dir + '/' + out_filename

      return out_dir, out_filename;

  @staticmethod
  def get_prfstats_filts_path():
      
      suffix = randomness.get_suffix();
      base = 'prfstats/data/' + suffix;

      try:
          os.makedirs(base);
      except OSError:
          pass

      return (base + '/filts.npy')
      
