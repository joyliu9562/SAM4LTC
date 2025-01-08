This is the official implementation of SAM4LTC

Here we provide:

- Source code for SAM4LTC in `model.py`
- Dependency utils in `utils.py`
- Manual annotated data in `/data/`
- Use `data_loader.py` to load data
- For training and testing use `train_test.py`
- For LLM refine, see `LLM_refine.py`

For the data analysis part

* data of international migrations for people from Germany and the United States is in `analysis/data/`
* data analysis code is in `analysis`

## Abstract

Life trajectories of notable people convey essential messages for the wide work on human dynamics. These trajectories consist of (person, time, location, activity type) tuples, and may record when and where a person was born, passed away, went to school, started a job, got married, made a scientific discovery, finished a masterpiece, or won an election. However, current studies only cover very limited types of activities, such as births and deaths, which are easier to obtain -- there lack large-scale trajectories covering fine-grained activity types. After extracting (person, time, location) triples from Wikipedia with an existing tool, we formulate a problem of classifying these triples into 24 carefully-defined types, given the triples' textual contexts as complementary information. Apart from the difficulty of multi-classification, it is challenging since the entities consisting of the triples are often scattered afar in the context, with plenty of noise around. To better emphasize and aggregate the semantic relations between the focal triple entities  and their related text, we make use of the syntactic graphs that bring the triple entities and their relevant information closer, and fuse them with the contextual embeddings to classify life trajectory activities. Meanwhile, we use LLM to refine the text from the crowd-sourced Wikipedia, resulting in more standard and unified syntactic graphs. Our framework outperforms baselines with an accuracy of 85.0%. With fine-grained trajectories of 589,193 people, we touch on grand narratives of human activities spanning 3 centuries. To facilitate relevant research, we make the code, the manually-labeled dataset, and the 3.8 million trajectory triples with classifications publicly available.

## Requirements

python >= 3.8.0

See requirements.txt