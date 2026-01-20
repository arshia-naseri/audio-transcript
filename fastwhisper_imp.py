# For later implementation and testing
from faster_whisper import WhisperModel
import functools
from time import perf_counter


#? Helper function to get the runtime
def timeit(fcn):
    @functools.wraps(fcn)
    def wrapper(*args,**kwargs):
        start_time = perf_counter()
        result = fcn(*args, **kwargs)
        end_time = perf_counter()
        runtime = end_time - start_time
        name = kwargs.get("model_name", args[0] if args else "unknown")
        print(f"{name} executed in {runtime:.6f} seconds")
        return result
    return wrapper

@timeit
def getTranscript(model_name:str, model, audio_path:str, language:str):
    segments, info = model.transcribe(audio_path, beam_size=5,language = language)
    for seg in segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")

model = WhisperModel(
    # "large-v3",
    "large-v3-turbo",
    device="cpu",          # CPU #metal
    compute_type="int8",    # good speed on CPU, slightly less accurate
    download_root="./models",
    local_files_only=False  # Downloads if missing, uses cached if exists
)

'''
common compute_type choices on CPU:
"int8": fastest on most CPUs (good default)
"int8_float16": sometimes better quality than pure int8 (fastest on GPU)
"float32": slowest, highest numeric precision (for gpu)
'''
audio_path = "./audios/CH_Where_is_toronto.m4a"

getTranscript("large-v3",model,audio_path,"zh")