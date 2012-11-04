package withPerceptronClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import withPerceptronClassifier.argumentClassifier.ArgumentClassifier;
import withPerceptronClassifier.argumentClassifier.ArgumentClassifierA;
import withPerceptronClassifier.argumentClassifier.ArgumentClassifierB;
import withPerceptronClassifier.argumentClassifier.ArgumentClassifierC;
import withPerceptronClassifier.argumentClassifier.ArgumentClassifierToken;
import withPerceptronClassifier.classify.PerceptronClassifier;
import withPerceptronClassifier.predicateClassifier.FeaturedPredicateToken;
import withPerceptronClassifier.predicateClassifier.PredicateClassifier;

public class Main {

	/**
	 * @param args
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 */
	public static final boolean USE_PREDICTED_PREDICATES = true;
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		
		int i = 0;
		List<List<FeaturedPredicateToken>> list = PredicateClassifier.sentencesFromCorpus("Testing\\devel.closed");
		for (List<FeaturedPredicateToken> listCeption : list)
			for (FeaturedPredicateToken t : listCeption)
				if (t.goldIsPredicate())
					i++;
		System.out.println(i);
		
		
		
		
		PerceptronClassifier classifier = new PerceptronClassifier();
		//Dataset<String, String> trainSet = ArgumentClassifierA.dataSetFromCorpus("Testing\\minitest.closed");
		//Dataset<String, String> trainSet = ArgumentClassifierA.dataSetFromCorpus("Testing\\train.closed");
		//trainSet.applyFeatureCountThreshold(3);
		
		//classifier.train(trainSet);
		//classifier.save("PerceptronTesting\\argumentClassifierA.gz");
		
		List<List<ArgumentClassifierToken>> sentences = ArgumentClassifier.sentencesFromCorpus("Testing\\devel.closed");		
		PredicateClassifier predicateClassifier;
		
		classifier = PerceptronClassifier.load("PerceptronTesting\\argumentClassifierA.gz");
		ArgumentClassifierC argumentClassifier = new ArgumentClassifierC(classifier);
		
		
		Counter<String> aPredicted = new ClassicCounter<String>();
		Counter<String> aCorrect = new ClassicCounter<String>();
		Counter<String> goldLabels = new ClassicCounter<String>();
		
		if (USE_PREDICTED_PREDICATES)
			predicateClassifier = new PredicateClassifier(PerceptronClassifier.load("PerceptronTesting\\predicateClassifierA'.gz"));
		
		for (List<ArgumentClassifierToken> sentence : sentences){
			
			List<ArgumentClassifierToken> predicates;
			List<ArgumentClassifierToken> goldPredicates = (List<ArgumentClassifierToken>) PredicateClassifier.goldPredicatesInSentence(sentence);
			
			if (USE_PREDICTED_PREDICATES)
				predicates = (List<ArgumentClassifierToken>) predicateClassifier.predicatesInSentence(sentence);
			else
				predicates = goldPredicates;
			
			for (ArgumentClassifierToken predicate : predicates){
				
				Map<ArgumentClassifierToken, String> aArgumentLabels = argumentClassifier.argumentsOf(predicate);
				Map<ArgumentClassifierToken, String> goldArgumentLabels = ArgumentClassifier.goldArgumentsOf(predicate);
				
				for (ArgumentClassifierToken argument : ArgumentClassifier.argumentCandidates(predicate)){
					String aPredictedLabel = aArgumentLabels.get(argument);
					String goldLabel = goldArgumentLabels.get(argument);
					
					aPredicted.incrementCount(aPredictedLabel);
					goldLabels.incrementCount(goldLabel);
					if (aPredictedLabel.equals(goldLabel))
						aCorrect.incrementCount(goldLabel);
				}
			}
		}
		
		List<String> argClasses = new ArrayList<String>();
		argClasses.addAll(goldLabels.keySet());
		Collections.sort(argClasses);
		
		int totalCorrect = 0;
		int totalPredicted = 0;
		int totalGold = 0;
		for(String label : argClasses){
			totalCorrect += aCorrect.getCount(label);
			totalPredicted += aPredicted.getCount(label);
			totalGold += goldLabels.getCount(label);
			/*System.out.println(label + '\t' +
					(int) aCorrect.getCount(label) + '\t' +
					(int) aPredicted.getCount(label) + '\t' +
					(int) goldLabels.getCount(label));*/
		}
		System.out.println("Overall\t" + totalCorrect + "\t" + totalPredicted + "\t" + totalGold);
		System.out.println("Overall -NIL\t" + (int) (totalCorrect - aCorrect.getCount("NIL"))
				+ "\t" + (int) (totalPredicted - aPredicted.getCount("NIL"))
				+ "\t" + (int) (totalGold - goldLabels.getCount("NIL")));
		
	}

}
