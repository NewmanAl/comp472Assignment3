https://github.com/NewmanAl/comp472Assignment3

main.py is the main entry point for this assignment. It can be executed as follows
```
python main.py
```
main assumes that the working directory contains the training and test data, covid_training.tsv and covid_test_public.tsv. During execution, the files eval_NB-BOW-OV.txt, eval_NB-BOW-FV.txt, trace_NB-BOW-OV.txt and trace_NB-BOW-FV.txt will be generated.

generateDiagrams.py is a utility script to create the diagrams used for the assignment presentation. It can also be executed as follows
```
python generateDiagrams.py
```
generateDiagrams assumes that the working directory contains the output of both NB-BOW models and output for four different runs of the LSTM model. From the working directory, generateDiagram assumes the following file structure.
```
.
├── eval_NB-BOW-OV.txt
├── eval_NB-BOW-OV.txt
├── run1
│   └── eval_lstm.txt
├── run2
│   └── eval_lstm.txt
├── run3
│   └── eval_lstm.txt
└── run4
    └── eval_lstm.txt
```
The generated diagrams are placed within a created "diagrams" directory relative to the working directory.