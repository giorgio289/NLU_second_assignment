# NLU_second_assignment
## Requirements:
To run the code you need at least Python 2.7 and [SpaCy](https://spacy.io/)<br>
To install SpaCy run:<br>
```pip install spacy``` or ```conda install -c conda-forge spacy```<br>
Download english language with:<br>
```python -m spacy download en_core_web_sm```<br>
## Repo structure
The repo contains the funciotns required for the assignment in main.py and the provided conll.py functions, in the data directory there are 3 text files in CoNLL format.
## Short code description
The code available at [main.py](https://github.com/giorgio289/NLU_second_assignment/blob/main/main.py) provides both the required functions and the test function.
### Functions:
There are several functions which are required to execute the main ones which are:
* ```evaluate_NER_on_CoNLL(path)``` which evaluates the performance of SpaCy NER on CoNLL 2003 data available in [data](https://github.com/giorgio289/NLU_second_assignment/tree/main/data) folder. Takes as input the path of the conll file and returns a tuple of table for class level performance and token accuracy
* ```group_entities(text)``` which groups entity using noun_chunk and compute the frequency of each touple found in the input text. Takes as input a text as a string a returns a tuple of the generated list of list of grouped labels and the dictionary with relative and absolute frequency of each tuple  
* ```fix_segmentation(text)``` which fixes segmentation errors extending chunks based on compound dependency. Takes as input a text as a string and returns a list list of tuples of the form (text,NER_tag)
### Test function:
Allows the test of the main funcine and prints result in a pretty way. After setting the path of the CoNLL file and the text it prints the results
### Note:
For a more detailed description of the code see [report.pdf](https://github.com/giorgio289/NLU_first_assignment/blob/main/report.pdf)
