import keyboard
from just_playback import Playback

playback = Playback()
playback.stop()
playback.load_file('Example_Audio.mp3')
playback.play()


while True:
    if(keyboard.read_key == "q"):
        playback.pause()