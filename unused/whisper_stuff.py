import whisper
model = whisper.load_model("base")

audio = "/home/thrall/Videos/nsl.webm"
transcribe = model.transcribe(audio)
# translate = model.transcribe(audio, task="translate")

print(transcribe["text"])