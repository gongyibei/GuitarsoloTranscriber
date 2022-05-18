# GuitarsoloTranscriber

transcribe guitar solo audio to midi-like tab.

# Dataset

The training dataset used in this project is [GuitarSet](https://guitarset.weebly.com/).

# Result
test with audio file, and generate the result figure.

```bash
python3 ./test.py
```

![solo result](https://raw.githubusercontent.com/gongyibei/GuitarsoloTranscriber/master/assets/solo_result.png)

![comp result](https://raw.githubusercontent.com/gongyibei/GuitarsoloTranscriber/master/assets/comp_result.png)

# Transcribe

run the transcriber in streaming mode

```bash
parec --channels=1 --format=float32 --rate=44100 | python3 ./transcribe
```
