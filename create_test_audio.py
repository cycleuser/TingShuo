import wave
import struct

def create_silent_wav(filename, duration=1):
    sample_rate = 44100
    n_frames = int(sample_rate * duration)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        data = struct.pack('<' + ('h'*n_frames), *([0]*n_frames))
        wav_file.writeframes(data)

if __name__ == "__main__":
    create_silent_wav("test_audio.wav", duration=2)
