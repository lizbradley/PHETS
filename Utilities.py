from memory_profiler import profile

def mem_profile(f, flag):
	if flag: return profile(stream=f)

	else: return lambda x: x