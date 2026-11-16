# Add INCLUDE words to dynamic recognition

The existing static dataset, captured dynamic dataset, and trained model files are
left in place. INCLUDE-derived sequences are stored separately in
`dataset/include_raw` and `dataset/include_dynamic`; the expanded model is saved
as `models/isl_word_bilstm_include.pth` with labels in
`models/word_labels_include.json`.

The INCLUDE Zenodo record contains 4,292 videos across 263 signs. Downloading
both default categories (Greetings and Colours) is about 5 GB. The downloader
shows the exact size and asks for confirmation before starting.

From the project root, run:

```powershell
python module1_sign_to_speech\include_dynamic_dataset.py download
python module1_sign_to_speech\include_dynamic_dataset.py extract
python module1_sign_to_speech\train_include_word_model.py
python module1_sign_to_speech\live_word_prediction.py
```

Choose other categories (such as `Days_and_Time`, `Pronouns`, or `Home`) with:

```powershell
python module1_sign_to_speech\include_dynamic_dataset.py download --categories Greetings Days_and_Time Pronouns
```

The video extractor uses the same two-hand, 21-landmark, wrist-normalized
features as the existing dynamic collector and resamples each video to 30 frames.
It uses a word label from the INCLUDE split CSV when available, or from a
word-named directory. Videos without an identifiable word label or with too few
hand detections are reported and skipped. If no sequences are produced, inspect
the extracted INCLUDE folder layout and split CSV before training.

Training combines valid INCLUDE sequences with the existing captured dynamic
sequences. It writes a new model only; live recognition prefers that model if
both new model files exist and otherwise continues using the original model.
Review the Zenodo record for dataset citation and use terms before redistribution:
<https://zenodo.org/records/4010759>.
