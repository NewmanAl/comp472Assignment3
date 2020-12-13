# COMP472 Assignment 3
# Alexander Newman
# 27021747
# Dec 2020

from main import Word
from main import generateVocabulary
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
import numpy as np

trainingFile = "covid_training.tsv"
testFile = "covid_test_public.tsv"

def main():
  #ensure "diagrams" directory exists
  Path("diagrams").mkdir(parents=True, exist_ok=True)
  
  #data on vocabulary
  vocab, totalFactualTweets, totalNotFactualTweets = generateVocabulary(trainingFile)
  print("Training factual tweets: " + str(totalFactualTweets))
  print("Training nonfactual tweets: " + str(totalNotFactualTweets))
  
  lenOV = len(vocab)
  numFactualWordsOV = 0
  numNotFactualWordsOV = 0
  
  for word in vocab:
    numFactualWordsOV += vocab[word].numFactual
    numNotFactualWordsOV += vocab[word].numNotFactual
    
  #filter vocabulary
  for key in list(vocab.keys()):
    if vocab[key].numFactual + vocab[key].numNotFactual < 2:
      del vocab[key]
      
  lenFV = len(vocab)
  numFactualWordsFV = 0
  numNotFactualWordsFV = 0
  
  for word in vocab:
    numFactualWordsFV += vocab[word].numFactual
    numNotFactualWordsFV += vocab[word].numNotFactual
    
  #vocabulary chart
  OVData = [lenOV, numFactualWordsOV, numNotFactualWordsOV]
  FVData = [lenFV, numFactualWordsFV, numNotFactualWordsFV]
  
  x = np.arange(3)
  width = 0.35
  
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, OVData, width, label="Original Vocab")
  rects2 = ax.bar(x + width/2, FVData, width, label="Filtered Vocab")
  
  ax.set_title("Original vs Filtered Vocabulary")
  ax.set_xticks(x)
  ax.set_xticklabels(["Vocab Length","Occurrences of\nFactual Words","Occurrences of\nNonfactual Words"])
  ax.legend()
  
  for i in range(3):
    plt.text(i - width/2, OVData[i]+0.003, str(OVData[i]), horizontalalignment='center')
    plt.text(i + width/2, FVData[i]+0.003, str(FVData[i]), horizontalalignment='center')
  
  fig.tight_layout()
  #plt.show()
  plt.savefig("diagrams/vocabBreakdown.png")
  plt.close()
  
  #test data
  numTestFactual = 0
  numTestNotFactual = 0
  
  with open(testFile, "r", encoding="utf-8") as f:
    for line in f:
      if len(line) > 0:
        if line.split("\t")[2] == "yes":
          numTestFactual += 1
        else:
          numTestNotFactual += 1
          
  print("Test data factual tweets: " + str(numTestFactual))
  print("Test data nonfactual tweets: " + str(numTestNotFactual))
  
  #naive-bayes
  nbOvAccuracy, \
  nbOvYesPrecision, \
  nbOvNoPrecision, \
  nbOvYesRecall, \
  nbOvNoRecall, \
  nbOvYesF1, \
  nbOvNoF1 = readEvalFile("eval_NB-BOW-OV.txt")
  
  nbFvAccuracy, \
  nbFvYesPrecision, \
  nbFvNoPrecision, \
  nbFvYesRecall, \
  nbFvNoRecall, \
  nbFvYesF1, \
  nbFvNoF1 = readEvalFile("eval_NB-BOW-FV.txt")
  
  #accuracy
  fig, ax = plt.subplots()
  ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
  plt.bar(["NB-BOW-OV","NB-BOW-FV"], [nbOvAccuracy, nbFvAccuracy])
  plt.title("Naive-Bayes Accuracy")
  for index, value in enumerate([nbOvAccuracy, nbFvAccuracy]):
    plt.text(index, value + 0.003, '{0:.2%}'.format(value), horizontalalignment='center')
  #plt.show()
  plt.savefig("diagrams/nbAccuracy.png")
  plt.close()
  
  #ov performance
  yesData = [nbOvYesPrecision, nbOvYesRecall, nbOvYesF1]
  noData = [nbOvNoPrecision, nbOvNoRecall, nbOvNoF1]  
  performanceDiagram(yesData, noData, "NB-BOW-OV Performance", "diagrams/nbOvPerformance.png")

  #fv performance
  yesData = [nbFvYesPrecision, nbFvYesRecall, nbFvYesF1]
  noData = [nbFvNoPrecision, nbFvNoRecall, nbFvNoF1]
  performanceDiagram(yesData, noData, "NB-BOW-FV Performance", "diagrams/nbFvPerformance.png")
  
  #LSTM
  run1Accuracy, \
  run1YesPrecision, \
  run1NoPrecision, \
  run1YesRecall, \
  run1NoRecall, \
  run1YesF1, \
  run1NoF1 = readEvalFile("run1/eval_lstm.txt")
  
  run2Accuracy, \
  run2YesPrecision, \
  run2NoPrecision, \
  run2YesRecall, \
  run2NoRecall, \
  run2YesF1, \
  run2NoF1 = readEvalFile("run2/eval_lstm.txt")
  
  run3Accuracy, \
  run3YesPrecision, \
  run3NoPrecision, \
  run3YesRecall, \
  run3NoRecall, \
  run3YesF1, \
  run3NoF1 = readEvalFile("run3/eval_lstm.txt")
  
  run4Accuracy, \
  run4YesPrecision, \
  run4NoPrecision, \
  run4YesRecall, \
  run4NoRecall, \
  run4YesF1, \
  run4NoF1 = readEvalFile("run4/eval_lstm.txt")
  
  #accuracy
  fig, ax = plt.subplots()
  ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
  plt.bar(["Run 1","Run 2", "Run 3", "Run 4"], [run1Accuracy, run2Accuracy, run3Accuracy, run4Accuracy])
  plt.title("LSTM Accuracy")
  for index, value in enumerate([run1Accuracy, run2Accuracy, run3Accuracy, run4Accuracy]):
    plt.text(index, value + 0.003, '{0:.2%}'.format(value), horizontalalignment='center')
  #plt.show()
  plt.savefig("diagrams/lstmAccuracy.png")
  plt.close()
  
  #run1 performance
  yesData = [run1YesPrecision, run1YesRecall, run1YesF1]
  noData = [run1NoPrecision, run1NoRecall, run1NoF1]
  performanceDiagram(yesData, noData, "LSTM Run 1 Performance", "diagrams/lstmRun1Performance.png")
  
  #run2 performance
  yesData = [run2YesPrecision, run2YesRecall, run2YesF1]
  noData = [run2NoPrecision, run2NoRecall, run2NoF1]
  performanceDiagram(yesData, noData, "LSTM Run 2 Performance", "diagrams/lstmRun2Performance.png")
  
  #run3 performance
  yesData = [run3YesPrecision, run3YesRecall, run3YesF1]
  noData = [run3NoPrecision, run3NoRecall, run3NoF1]
  performanceDiagram(yesData, noData, "LSTM Run 3 Performance", "diagrams/lstmRun3Performance.png")
  
  #run4 performance
  yesData = [run4YesPrecision, run4YesRecall, run4YesF1]
  noData = [run4NoPrecision, run4NoRecall, run4NoF1]
  performanceDiagram(yesData, noData, "LSTM Run 4 Performance", "diagrams/lstmRun4Performance.png")
  

def readEvalFile(fileName):
  accuracy = 0
  yesPrecision = 0
  noPrecision = 0
  yesRecall = 0
  noRecall = 0
  yesF1 = 0
  noF1 = 0
  
  with open(fileName, "r") as f:
    lines = f.readlines()
    
    accuracy = float(lines[0])
    yesPrecision = float(lines[1].split()[0])
    noPrecision = float(lines[1].split()[1])
    yesRecall = float(lines[2].split()[0])
    noRecall = float(lines[2].split()[1])
    yesF1 = float(lines[3].split()[0])
    noF1 = float(lines[3].split()[1])
    
  return accuracy, yesPrecision, noPrecision, yesRecall, noRecall, yesF1, noF1

def performanceDiagram(yesData, noData, title, fileName):
  x = np.arange(3)
  width = 0.35
  
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, yesData, width, label="yes")
  rects2 = ax.bar(x + width/2, noData, width, label="no")
  
  ax.set_title(title)
  ax.set_xticks(x)
  ax.set_xticklabels(["Precision","Recall","F1-Score"])
  ax.yaxis.set_major_formatter(PercentFormatter(xmax = 1.0))
  ax.legend()
  
  for i in range(3):
    plt.text(i - width/2, yesData[i]+0.003, '{0:.2%}'.format(yesData[i]), horizontalalignment='center')
    plt.text(i + width/2, noData[i]+0.003, '{0:.2%}'.format(noData[i]), horizontalalignment='center')
  
  fig.tight_layout()
  #plt.show()
  plt.savefig(fileName)
  plt.close()

if __name__ == '__main__':
  main()