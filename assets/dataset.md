SQA3D data
===

## Guide

1. Download the [SQA3D dataset](https://zenodo.org/record/7792397/files/sqa_task.zip?download=1) under `assets/data/`. The following files should be used:
```plain
./assets/data/sqa_task/balanced/*
./assets/data/sqa_task/answer_dict.json
```

2. The dataset has been splited into `train`, `val` and `test`. For each category, we offer both question file, ex. `v1_balanced_questions_train_scannetv2.json`, and annotations, ex. `v1_balanced_sqa_annotations_train_scannetv2.json`

- The format of question file:

  Run the following code:
  ```python
  import json
  q = json.load(open('v1_balanced_questions_train_scannetv2.json', 'r'))
  # Print the total number of questions
  print('#questions: ', len(q['questions']))
  print(q['questions'][0])
  ```
  The output is:
  ```json
  {
    "alternative_situation":
      [
        "I stand looking out of the window in thought and a radiator is right in front of me.",
        "I am looking outside through the window behind the desk."
      ],
    "question": "What color is the desk to my right?",
    "question_id": 220602000000,
    "scene_id": "scene0380_00",
    "situation": "I am facing a window and there is a desk on my right and a chair behind me."
  }
  ```
  The following fileds are **useful**: `question`, `question_id`, `scene_id`, `situation`.

- The format of annotations:

  Run the following code:
  ```python
  import json
  a = json.load(open('v1_balanced_sqa_annotations_train_scannetv2.json', 'r'))
  # Print the total number of annotations, should be the same as questions
  print('#annotations: ', len(a['annotations']))
  print(a['annotations'][0])
  ```
  The output is
  ```json
  {
    "answer_type": "other",
    "answers":
      [
        {
          "answer": "brown",
          "answer_confidence": "yes",
          "answer_id": 1
        }
      ],
    "position":
      {
        "x": -0.9651003385573296,
        "y": -1.2417634435553606,
        "z": 0
      },
    "question_id": 220602000000,
    "question_type": "N/A",
    "rotation":
      {
        "_w": 0.9950041652780182,
        "_x": 0,
        "_y": 0,
        "_z": 0.09983341664682724
      },
    "scene_id": "scene0380_00"
  }
  ```
  The following fields are **useful**: `answers[0]['answer']`, `question_id`, `scene_id`.

  **Note**: To find the answer of a question in the question file, you need to use lookup with `question_id`.

3. We provide the mapping between answers and class labels in `answer_dict.json`
```python
import json
j = json.load(open('answer_dict.json', 'r'))
print('Total classes: ', len(j[0]))
print('The class label of answer \'table\' is: ', j[0]['table'])
print('The corresponding answer of class 123 is: ', j[1]['123'])
```

4. To obtain the scene data, please refer to [3D scans for ScanQA](../ScanQA/README.md), [egocentric videos for ClipBERT](../ClipBERT/README.md) and [BEV pictures for MCAN](../MCAN/README.md) for more details.



## Benchmarking

See [bencmarking and leaderboard](./benchmarking_leaderboard.md)
