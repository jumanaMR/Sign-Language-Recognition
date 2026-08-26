# Sign Language Recognition

Hand-sign detection using a custom object detection model trained on Azure Custom
Vision and exported to TensorFlow for local inference.

## What this is

I built an object detection model that locates hand signs in an image and labels
them. The dataset was collected and annotated by me, trained through **Azure Custom
Vision**, then exported to the TensorFlow format so it runs locally with no cloud
call at inference time.

The classes the model recognises are listed in `labels.txt`, in the same index
order the model outputs.

This project was about the **end-to-end pipeline** rather than the model
architecture: collecting and labelling data, training, evaluating, exporting, and
wiring up local inference. The training loop and architecture selection are Azure's
AutoML; the dataset, the labelling, and the local inference path are mine.

## How it was built

1. **Data collection** — captured images of each hand sign, varying position and
   distance within a consistent indoor setup.
2. **Annotation** — drew bounding boxes and assigned class labels for every image
   in the Custom Vision portal.
3. **Training** — trained an object detection iteration, reviewed the per-class
   precision and recall the portal reports, and re-labelled the classes that were
   confusing the model before retraining.
4. **Export** — exported the trained iteration to TensorFlow, which produces the
   frozen graph plus the inference scaffolding in this repo.
5. **Local inference** — ran the exported model against held-out images to confirm
   it behaved the same outside the portal as it did inside it.

## Repository contents

| File | What it is |
|---|---|
| `predict.py` | Entry point — loads the model and runs detection on one image |
| `object_detection.py` | Inference wrapper: preprocessing, TF session, postprocessing |
| `model.pb` | Frozen TensorFlow graph containing the trained weights |
| `labels.txt` | Class names, in model output order |
| `cvexport.manifest` | Export metadata from Azure Custom Vision |
| `metadata_properties.json` | Model input/output specification |

`object_detection.py` and the surrounding inference scaffolding come from Azure's
TensorFlow export template rather than being written by hand.

## Running it

```bash
git clone https://github.com/jumanaMR/Sign-Language-Recognition.git
cd Sign-Language-Recognition
pip install -r requirements.txt
python predict.py path/to/image.jpg
```

The script prints each detection with its class label, confidence score, and
bounding box coordinates.

## Limits

Worth being clear about what this does and doesn't do:

- **It is not a sign language translator.** Sign languages are grammatical and
  continuous, with meaning carried by movement, facial expression and spatial
  reference. This detects static single-hand gestures from a fixed vocabulary.
  That's a much smaller problem, and conflating the two does a disservice to the
  people who actually use these languages.
- **Still images, not video.** `predict.py` runs on one image at a time. A
  real-time webcam loop isn't implemented here.
- **Narrow training conditions.** The data was captured in consistent lighting
  against a plain background, from a small number of hands. Accuracy will drop
  outside those conditions, and I'd expect it to drop sharply on skin tones and
  hand shapes underrepresented in the data I collected.
- **The architecture wasn't my decision.** Custom Vision's AutoML selected the
  model and hyperparameters. I can explain what the export contains and how
  inference works, but not why that particular architecture was chosen.

## What I'd do differently

Custom Vision was the wrong tool for the job I actually had. It optimises for
getting a working model fast without ML expertise, and I paid for that speed in
control I would have wanted — no access to the architecture, no ability to tune
the training loop, and an export I can run but can't meaningfully modify. For a
fixed, offline, single-purpose detector, that trade was backwards.

If I rebuilt this, I'd extract hand landmarks with MediaPipe and train a small
classifier on the keypoints instead of on raw pixels. Hand geometry is a far more
compact and stable representation than an image — the resulting model would be
orders of magnitude smaller, run comfortably in real time on CPU, and be
substantially more robust to background and lighting, because most of what varies
between two photos of the same sign is thrown away before the classifier ever sees
it.

I'd also change how I evaluated it. I relied on the platform's own train/test split
and its reported metrics, which meant my evaluation set came from the same session,
the same lighting, and the same hands as my training set. That measures memorisation
more than it measures generalisation. A held-out set captured separately — different
day, different lighting, different person — would have told me something real about
whether the model works.

## License

MIT
