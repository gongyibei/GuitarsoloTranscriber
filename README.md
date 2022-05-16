# GuitarsoloTranscriber

transcribe guitar solo audio to midi-like tab.

# Dataset

The training dataset used is [GuitarSet](https://guitarset.weebly.com/).

# Result

test wav file.

```bash
python3 ./test.py
```

![solo result](https://raw.githubusercontent.com/gongyibei/GuitarsoloTranscriber/master/assets/solo_result.png)

![comp result](https://raw.githubusercontent.com/gongyibei/GuitarsoloTranscriber/master/assets/comp_result.png)

# Transcribe

run the transcriber in streaming mode

```bash
parec --format=float32 --rate=44100 | python3 ./transcribe
```
