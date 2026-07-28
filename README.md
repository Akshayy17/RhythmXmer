# RhythmFormer: A Multi-Task Transformer Framework for Rhythmic Analysis of Hindustani Classical Music


This is a multi-task neural network for automatic rhythm analysis of
Hindustani (North Indian) classical tabla recordings. Given an audio
recording, it predicts:

- **Taal** — the rhythmic cycle (e.g., Teentaal, Ektaal, Rupak, Jhaptaal)
- **Musical scale** — the tonic/key of the recording
- **Tempo** — beats per minute
- **Cycle period** — duration in seconds of one complete taal cycle
- **Sam** — frame-level detection of the downbeat marking the start of each cycle


## Repository structure

| File | Purpose |
|---|---|
| `config.py` | All paths, hyperparameters, taal/scale vocabulary, loss weights |
| `extract_embeddings.py` | Preprocesses raw audio into cached MERT embeddings |
| `dataset.py` | PyTorch `Dataset`, label completion logic, batch collation |
| `model.py` | Transformer-based multi-task model |
| `model_gru.py` | GRU-based multi-task model (architecture ablation) |
| `train.py` | Training loop and masked multi-task loss |
| `eval.py` | Segment-level and recording-level evaluation, sam event metrics |
| `infer.py` | Runs the trained model end-to-end on a single audio file |



## To Infer
```bash
python infer.py /path/to/recording.wav --ckpt /path/to/model.pt --sam-threshold 0.5
```
