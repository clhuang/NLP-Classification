package withLinearClassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import withLinearClassifier.argumentClassification.ArgumentClassifier;
import withLinearClassifier.argumentClassification.ArgumentClassifierA;
import withLinearClassifier.argumentClassification.ArgumentClassifierB;
import withLinearClassifier.argumentClassification.ArgumentClassifierC;
import withLinearClassifier.argumentClassification.ArgumentClassifierToken;
import withLinearClassifier.predicateClassification.PredicateClassifier;

public class Main {
	
	public static final boolean USE_PREDICTED_PREDICATES = true;

	/**
	 * @param args
	 * @throws IOException 
	 */
	
	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws IOException {
		
		ArgumentClassifierC argumentClassifierC;
		LinearClassifier<String, String> l;
		PredicateClassifier predicateClassifier;
		
		/*Dataset<String, String> trainSet = ArgumentClassifier.dataSetFromCorpus("Testing\\train.closed");
		trainSet.applyFeatureCountThreshold(3);
		l = new LinearClassifierFactory<String, String>().
				trainClassifier(trainSet);
		LinearClassifier.writeClassifier(l, "argumentclassifierB.gz");*/
		l  = LinearClassifier.readClassifier("Testing\\argumentclassifierA.gz");
		argumentClassifierC = new ArgumentClassifierC(l);
		
		if (USE_PREDICTED_PREDICATES){
			LinearClassifier<String, String> pl;
			pl = LinearClassifier.readClassifier("Testing\\predicateclassifierA.gz");
			predicateClassifier = new PredicateClassifier(pl);
		}
		List<List<ArgumentClassifierToken>> sentences = ArgumentClassifier.sentencesFromCorpus("Testing\\devel.closed");
		
		System.out.println();
		
		int predictedPredicates = 0;
		int correctPredicates = 0;
		int goldPredicateCount = 0;
		
		Counter<String> aCorrect = new ClassicCounter<String>();
		Counter<String> aPredicted = new ClassicCounter<String>();
		
		for (List<ArgumentClassifierToken> sentence : sentences){
			List<ArgumentClassifierToken> predicates;
			List<ArgumentClassifierToken> goldPredicates = (List<ArgumentClassifierToken>) PredicateClassifier.goldPredicatesInSentence(sentence);
			if (USE_PREDICTED_PREDICATES)
				predicates = (List<ArgumentClassifierToken>) predicateClassifier.predicatesInSentence(sentence);
			else
				predicates = goldPredicates;
			
			predictedPredicates += predicates.size();
			goldPredicateCount += goldPredicates.size();
			
			for (ArgumentClassifierToken predicate : predicates){
				
				if (goldPredicates.contains(predicate))
					correctPredicates++;
				
				Map<ArgumentClassifierToken, String> aArgumentLabels = argumentClassifierC.argumentsOf(predicate);
				Map<ArgumentClassifierToken, String> goldArgumentLabels = ArgumentClassifier.goldArgumentsOf(predicate);
				
				for (ArgumentClassifierToken argument : ArgumentClassifier.argumentCandidates(predicate)){
					String aPredictedLabel = aArgumentLabels.get(argument);
					String goldLabel = goldArgumentLabels.get(argument);
					
					aPredicted.incrementCount(aPredictedLabel);
					if (aPredictedLabel.equals(goldLabel))
						aCorrect.incrementCount(goldLabel);
				}
			}
		}
		
		List<String> argClasses = new ArrayList<String>();
		argClasses.addAll(aPredicted.keySet());
		Collections.sort(argClasses);
		
		System.out.println(correctPredicates + " " + predictedPredicates + " " + goldPredicateCount);
		
		for(String label : argClasses){
			System.out.println(label + '\t' +
					(int) aCorrect.getCount(label) + '\t' +
					(int) aPredicted.getCount(label));
		}
		
		/*for (String argClass : argClasses){
			int[] stat = stats.get(argClass);
			double precision = ((double) stat[2]) / ((double) stat[0]);
			double recall = ((double) stat[2]) / ((double) stat[1]);
			double f1 = 2 / ((1 / precision) + (1 / recall));
			
			if (Double.isNaN(precision))
				precision = 0;
			if (Double.isNaN(recall))
				recall = 0;
			if (Double.isNaN(f1))
				f1 = 0;
			
			System.out.format("%s %d %d %d %.3f %.3f %.3f%n",
					argClass,
					stat[0],
					stat[1],
					stat[2],
					precision,
					recall,
					f1);
		}*/
		
		//List<ArgumentClassifierToken> predicates = (List<ArgumentClassifierToken>) predicateClassifier.predicatesInSentence(sentences.get(0));
		
		/*for (ArgumentClassifierToken predicate : predicates){
			for(ArgumentClassifierToken argument : ArgumentClassifier.argumentCandidates(predicate)){
				String argClass = argumentClassifier.argClass(argument, predicate);
				if (!argClass.equals("NIL"))
					System.out.println(predicate.splitForm + " " + argument.splitForm + " " + argClass);
			}
		}*/
		
		//predicateClassifierTest();
		
	}

	public static void predicateClassifierTest() throws NumberFormatException, IOException {
		PredicateClassifier predicateClassifier;
		LinearClassifier<String, String> l;
		
		l  = LinearClassifier.readClassifier("Testing\\predicateclassifierA.gz");
		
		
		
		/*l = new LinearClassifierFactory<String, String>().
				trainClassifier(PredicateClassifier.dataSetFromCorpus("Testing\\train.closed"));
		LinearClassifier.writeClassifier(l, "Testing\\predicateclassifierCREL.gz");*/

		Dataset<String, String> dataset = PredicateClassifier.dataSetFromCorpus("Testing\\devel.closed");
		predicateClassifier = new PredicateClassifier(l);
		
		int[] nounStats = new int[4];
		int[] verbStats = new int[4];
		int[] generalStats = new int[4];
		int statPosition;
		
		final int TRUE_POSITIVE = 0;
		final int TRUE_NEGATIVE = 1;
		final int FALSE_POSITIVE = 2;
		final int FALSE_NEGATIVE = 3;
		
		for (Datum<String, String> d : dataset){
			String label = d.label();
			String pos = "";
			
			for (String s : d.asFeatures()){
				if (s.startsWith("pposu,i|")){
					pos =  s.substring(8);
					break;
				}
			}
			
			
			if (label.equals("predicate")){
				if (predicateClassifier.isPredicate(d))
					statPosition = TRUE_POSITIVE;
				else
				{
					ArrayList<String> splitLemma = new ArrayList<String>();
					for (String s : d.asFeatures()){
						if (s.startsWith("splmu"))
							splitLemma.add(s);
					}
					statPosition = FALSE_NEGATIVE;
				}
			}
			else{
				if (predicateClassifier.isPredicate(d)){
					ArrayList<String> splitLemma = new ArrayList<String>();
					for (String s : d.asFeatures()){
						if (s.startsWith("splmu"))
							splitLemma.add(s);
					}
					statPosition = FALSE_POSITIVE;
				}
					
					
				else
					statPosition = TRUE_NEGATIVE;
				
			}
			
			generalStats[statPosition]++;
			if (pos.startsWith("NN"))
				nounStats[statPosition]++;
			else if (pos.startsWith("VB"))
				verbStats[statPosition]++;

		}

		System.out.println();
		System.out.println("Noun stats:");
		for (int i : nounStats)
			System.out.println(i);
		
		System.out.println();
		System.out.println("Verb stats:");
		for (int i : verbStats)
			System.out.println(i);
		
		System.out.println();
		System.out.println("General stats:");
		for (int i : generalStats)
			System.out.println(i);
		

	}

}
