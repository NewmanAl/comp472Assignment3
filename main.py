# COMP472 Assignment 3
# Alexander Newman
# 27021747
# Dec 2020

from math import log10

trainingFileName = "covid_training.tsv"
testFileName = "covid_test_public.tsv"

class Word:
  numFactual = 0
  numNotFactual = 0

def main():
  vocab, totalFactualTweets, totalNotFactualTweets = generateVocabulary(trainingFileName)
  
  print("Running NB-BOW-OV")
  runModel(testFileName, "NB-BOW-OV", vocab, totalFactualTweets, totalNotFactualTweets)
  analyseOutput("NB-BOW-OV")
  
  #filter vocabulary and rerun
  for key in list(vocab.keys()):
    if vocab[key].numFactual + vocab[key].numNotFactual < 2:
      del vocab[key]
      
  print("Running NB-BOW-FV")
  runModel(testFileName, "NB-BOW-FV", vocab, totalFactualTweets, totalNotFactualTweets)
  analyseOutput("NB-BOW-FV")

def generateVocabulary(fileName):
  vocab = {}
  totalFactualTweets = 0
  totalNotFactualTweets = 0
  with open(fileName, encoding="utf-8") as f:
    #skip first line (header line)
    next(f)
    for line in f:
      components = line.split("\t")
      if components[2] == "no":
        totalNotFactualTweets += 1
      else:
        totalFactualTweets += 1
        
      text = components[1].casefold().split()
      for word in text:
        if word not in vocab:
          vocab[word] = Word()
          
        if components[2] == 'no':
          vocab[word].numNotFactual += 1
        else:
          vocab[word].numFactual += 1

  return vocab, totalFactualTweets, totalNotFactualTweets

def runModel(inputFile, modelName, vocab, totalFactualTweets, totalNotFactualTweets):
  delta = 0.01
  
  #calculate totals from vocabulary
  sizeOfVocab = len(vocab)
  totalFactualWords = 0
  totalNotFactualWords = 0
  
  for word in vocab:
    totalFactualWords += vocab[word].numFactual
    totalNotFactualWords += vocab[word].numNotFactual
    
  outputStr = ""
  
  with open(inputFile, encoding="utf-8") as f:
    for line in f:
      probFactual = log10(totalFactualTweets / (totalFactualTweets + totalNotFactualTweets))
      probNotFactual = log10(totalNotFactualTweets / (totalFactualTweets + totalNotFactualTweets))
      components = line.split("\t")
      text = components[1].casefold().split()
      
      yesProb = probFactual
      noProb = probNotFactual
      
      for word in text:
        if word in vocab:
          probFactual += log10((vocab[word].numFactual + delta)/(totalFactualWords + (sizeOfVocab * delta)))
          probNotFactual += log10((vocab[word].numNotFactual + delta)/(totalNotFactualWords + (sizeOfVocab * delta)))
          
      isFactual = False
      if probFactual > probNotFactual:
        isFactual = True
      
      outputStr += components[0] + "  " + \
                   ("yes" if isFactual else "no") + "  " + \
                   format((probFactual if isFactual else probNotFactual), "e") + "  " + \
                   components[2] + "  " + \
                   ("correct" if (isFactual and components[2]=="yes") or (not isFactual and components[2]=="no") else "wrong") + "\n"
  
  #write results to file
  with open("trace_" + modelName + ".txt", "w") as f:
    f.write(outputStr)

def analyseOutput(modelName):
  yesTP, yesFP, yesFN, yesTN = 0,0,0,0
  noTP, noFP, noFN, noTN = 0,0,0,0
  totalLines = 0
  
  
  with open("trace_" + modelName + ".txt", "r") as f:
    for line in f:
      if len(line) > 0:
        totalLines += 1
        components = line.split()
        if components[1] == "yes" and components[3] == "yes":
          yesTP += 1
          noTN += 1
          
        if components[1] == "yes" and components[3] == "no":
          yesFP += 1
          noFN += 1
          
        if components[1] == "no" and components[3] == "yes":
          yesFN += 1
          noFP += 1
          
        if components[1] == "no" and components[3] == "no":
          yesTN += 1
          noTP += 1

  accuracy = (yesTP + noTP)/totalLines
  yesPrecision = yesTP/(yesTP + yesFP)
  noPrecision = noTP/(noTP + noFP)
  yesRecall = yesTP/(yesTP + yesFN)
  noRecall = noTP/(noTP + noFN)
  yesF1 = (2*yesPrecision*yesRecall)/(yesPrecision+yesRecall)
  noF1 = (2*noPrecision*noRecall)/(noPrecision+noRecall)
  
  with open("eval_" + modelName + ".txt", "w") as f:
    f.write(str(accuracy) + "\n")
    f.write(str(yesPrecision) + "  " + str(noPrecision) + "\n")
    f.write(str(yesRecall) + "  " + str(noRecall) + "\n")
    f.write(str(yesF1) + "  " + str(noF1) + "\n")
  
if __name__ == '__main__':
  main()