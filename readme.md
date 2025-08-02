# Audio Bug

## How to use

```
git clone https://github.com/Josh-Reimer/audiobug.git
cd audiobug
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run ```python3 pipewire_recorder.py``` to record audio based on decibal levels

Run ```python3 decibal_game.py``` to view current decibals in a graph

Run ```python3 realtime_human_voice_detector.py``` to detect human voices in realtime

Run ```python3 human_voice_detector.py``` to process files from ```pipewire_recorder.py``` for human voice occurence