package test;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import argumentClassification.ArgumentClassifier;
import argumentClassification.ArgumentClassifierA;
import argumentClassification.ArgumentClassifierB;
//import argumentClassification.ArgumentClassifier.ArgumentWithProbability;
import argumentClassification.ArgumentClassifierToken;

import predicateClassification.PredicateClassifier;
import util.CorpusUtils;

public class Main {

	/**
	 * @param args
	 * @throws IOException 
	 */
	
	public static void main(String[] args) throws IOException {
		ArgumentClassifierA argumentClassifierA;
		LinearClassifier<String, String> l;
		//LinearClassifier<String, String> pl;
		
		/*Dataset<String, String> trainSet = ArgumentClassifier.dataSetFromCorpus("train.closed");
		trainSet.applyFeatureCountThreshold(3);
		l = new LinearClassifierFactory<String, String>().
				trainClassifier(trainSet);
		LinearClassifier.writeClassifier(l, "argumentclassifierB.gz");*/
		
		l  = LinearClassifier.readClassifier("argumentclassifierA.gz");
		//pl = LinearClassifier.readClassifier("predicateclassifierA.gz");
		argumentClassifierA = new ArgumentClassifierA(l);
		//PredicateClassifier predicateClassifier = new PredicateClassifier(pl);
		
		LinearClassifier<String, String> l2 = LinearClassifier.readClassifier("argumentclassifierB.gz");
		ArgumentClassifierB argumentClassifierB = new ArgumentClassifierB(l2);
		
		List<List<ArgumentClassifierToken>> sentences = ArgumentClassifierB.sentencesFromCorpus("devel.closed");
		
		System.out.println();
		
		Counter<String> aCorrect = new ClassicCounter<String>();
		Counter<String> bCorrect = new ClassicCounter<String>();
		Counter<String> aPredicted = new ClassicCounter<String>();
		Counter<String> bPredicted = new ClassicCounter<String>();
		Counter<String> goldLabels = new ClassicCounter<String>();
		
		for (List<ArgumentClassifierToken> sentence : sentences){
		//List<ArgumentClassifierToken> sentence = sentences.get(1);{
			@SuppressWarnings("unchecked")
			List<ArgumentClassifierToken> predicates = (List<ArgumentClassifierToken>) PredicateClassifier.goldPredicatesInSentence(sentence);
			for (ArgumentClassifierToken predicate : predicates){
				
				Map<ArgumentClassifierToken, String> aArgumentLabels = argumentClassifierA.argumentsOf(predicate);
				Map<ArgumentClassifierToken, String> bArgumentLabels = argumentClassifierB.argumentsOf(predicate);
				Map<ArgumentClassifierToken, String> goldArgumentLabels = ArgumentClassifier.goldArgumentsOf(predicate);
				
				for (ArgumentClassifierToken argument : ArgumentClassifier.argumentCandidates(predicate)){
					String aPredictedLabel = aArgumentLabels.get(argument);
					String bPredictedLabel = bArgumentLabels.get(argument);
					String goldLabel = goldArgumentLabels.get(argument);
					
					goldLabels.incrementCount(goldLabel);
					aPredicted.incrementCount(aPredictedLabel);
					bPredicted.incrementCount(bPredictedLabel);
					if (aPredictedLabel.equals(goldLabel))
						aCorrect.incrementCount(goldLabel);
					if (bPredictedLabel.equals(goldLabel))
						bCorrect.incrementCount(goldLabel);
					
					/*if(aPredicted.equals(goldLabel) && !bPredicted.equals(goldLabel)){
						System.err.println(predicate.splitForm + ' ' + argument.splitForm + ' ' + aPredicted + ' ' + bPredicted + ' ' + goldLabel);
						System.err.flush();
					}
					else{
						System.out.println(predicate.splitForm + ' ' + argument.splitForm + ' ' + aPredicted + ' ' + bPredicted + ' ' + goldLabel);
						System.out.flush();
					}*/
				}
			}
		}
		
		List<String> argClasses = new ArrayList<String>();
		argClasses.addAll(goldLabels.keySet());
		Collections.sort(argClasses);
		
		for(String label : argClasses){
			System.out.println(label + ' ' +
					(int) aCorrect.getCount(label) + ' ' +
					(int) bCorrect.getCount(label) + ' ' +
					(int) aPredicted.getCount(label) + ' ' +
					(int) bPredicted.getCount(label) + ' ' +
					(int) goldLabels.getCount(label));
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
		
		l  = LinearClassifier.readClassifier("predicateclassifierCREL.gz");
		
		
		
		/*l = new LinearClassifierFactory<String, String>().
				trainClassifier(PredicateClassifier.dataSetFromCorpus("train.closed"));
		LinearClassifier.writeClassifier(l, "predicateclassifierCREL.gz");*/

		Dataset<String, String> dataset = PredicateClassifier.dataSetFromCorpus("devel.closed");
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
					//System.out.println("FALSE- " + splitLemma.get(0) + " " + splitLemma.get(1) + " " + splitLemma.get(2) + " ");
					//System.out.println();
					//System.out.println();
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
					//System.out.println("FALSE+ " + splitLemma.get(0) + " " + splitLemma.get(1) + " " + splitLemma.get(2) + " ");
					//System.out.println();
					//System.out.println();
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
