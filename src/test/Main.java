package test;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.Datum;

import java.io.IOException;
import predicateClassification.PredicateClassifier;

public class Main {

	/**
	 * @param args
	 * @throws IOException 
	 */

	public static void main(String[] args) throws NumberFormatException, IOException {
		PredicateClassifier predicateClassifier;
		LinearClassifier<String, String> l = LinearClassifier.readClassifier("predicateclassifierCDIF.gz");
		
		predicateClassifier = new PredicateClassifier(l);

		/*predicateClassifier = new PredicateClassifier(new LinearClassifierFactory<String, String>().
				trainClassifier(
						PredicateClassifier.dataSetFromCorpus("train.closed")));
		predicateClassifier.writeClassifier("predicateclassifierCABS.gz");*/
		 
		Dataset<String, String> dataset = PredicateClassifier.dataSetFromCorpus("devel.closed");

		int[] nounStats = new int[4];
		int[] verbStats = new int[4];
		int[] generalStats = new int[4];
		int statPosition;
		
		final int TRUE_POSITIVE = 0;
		final int TRUE_NEGATIVE = 1;
		final int FALSE_POSITIVE = 2;
		final int FALSE_NEGATIVE = 3;
		
		for(Datum<String, String> d : dataset){
			String label = d.label();
			String pos = "";
			
			for(String s : d.asFeatures()){
				if(s.startsWith("pposu,i|")){
					pos =  s.substring(8);
					break;
				}
			}
			
			if(label.equals("predicate")){
				if(predicateClassifier.isPredicate(d))
					statPosition = TRUE_POSITIVE;
				else
					statPosition = FALSE_NEGATIVE;
			}
			else{
				if(predicateClassifier.isPredicate(d))
					statPosition = FALSE_POSITIVE;
				else
					statPosition = TRUE_NEGATIVE;
			}
			
			generalStats[statPosition]++;
			if(pos.startsWith("NN"))
				nounStats[statPosition]++;
			else if(pos.startsWith("VB"))
				verbStats[statPosition]++;

		}

		System.out.println();
		System.out.println("Noun stats:");
		for(int i : nounStats)
			System.out.println(i);
		
		System.out.println();
		System.out.println("Verb stats:");
		for(int i : verbStats)
			System.out.println(i);
		
		System.out.println();
		System.out.println("General stats:");
		for(int i : generalStats)
			System.out.println(i);
		

	}

}
