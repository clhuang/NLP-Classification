package withPerceptronClassifier.argumentClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import withPerceptronClassifier.classify.PerceptronClassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;

public class ArgumentClassifierC extends ArgumentClassifier{

	public ArgumentClassifierC(PerceptronClassifier classifier){
		super(classifier);
	}

	public static Collection<String> getFeatures(ArgumentClassifierToken predicate,
			ArgumentClassifierToken argument){
		
		if (!argument.getSentenceTokens().equals(predicate.getSentenceTokens()))
			return null;
		
		Collection<String> features = new ArrayList<String>();

		/*
		 * Feature 1: argument (and modifier), predicate split lemma, form; pposs
		 */
		features.add("argsplm|" + argument.splitLemma);
		features.add("argspfm|" + argument.splitForm);
		features.add("argppos|" + argument.pposs);
		features.add("predsplm|" + predicate.splitLemma);
		features.add("predspfm|" + predicate.splitForm);
		features.add("predppos|" + predicate.pposs);
		
		ArgumentClassifierToken pmod = argument.getPMOD();
		if (pmod != null){
			features.add("pmodsplm|" + pmod.splitLemma);
			features.add("pmodspfm|" + pmod.splitForm);
			features.add("pmodppos|" + pmod.pposs);
		}
		
		/*
		 * Feature 2: pposs/deprel for predicate children, children of predicate ancestor across VC/IM dependencies
		 */
		StringBuilder deprelFeature = new StringBuilder("predcdeprel|");
		StringBuilder ppossFeature = new StringBuilder("predcpposs|");
		for (ArgumentClassifierToken child : predicate.getChildren()){
			deprelFeature.append(child.deprel + " ");
			ppossFeature.append(child.pposs + " ");
		}
		features.add(deprelFeature.toString());
		features.add(ppossFeature.toString());
		
		deprelFeature = new StringBuilder("vcimdeprel|");
		ppossFeature = new StringBuilder("vcimpposs|");
		ArgumentClassifierToken vcimAncestor = predicate;
		while(vcimAncestor.deprel.equals("VC") || vcimAncestor.deprel.equals("IM"))
			vcimAncestor = (ArgumentClassifierToken) vcimAncestor.getParent();
		
		for (ArgumentClassifierToken ancestorChild : vcimAncestor.getChildren()){
			deprelFeature.append(" " + ancestorChild.deprel);
			ppossFeature.append(" " + ancestorChild.pposs);
			if (ancestorChild.equals(argument)){
				deprelFeature.append("a");
				ppossFeature.append("a");
			}
			else if (ancestorChild.equals(predicate)){
				deprelFeature.append("p");
				ppossFeature.append("p");
			}
		}
		features.add(deprelFeature.toString());
		features.add(ppossFeature.toString());
		
		
		
		/*
		 * Feature 3: dependency path
		 */
		StringBuilder path = new StringBuilder();
		
		/*
		 * Ancestor splits the path into two halves: 
		 * from the argument to the ancestor, the dependencies go upwards;
		 * from the predicate to the ancestor, the dependencies go downwards
		 */
		ArgumentClassifierToken ancestor = SentenceUtils.getCommonAncestor(argument, predicate);
		
		Deque<ArgumentClassifierToken> argPath = SentenceUtils.ancestorPath(argument, ancestor);
		Deque<ArgumentClassifierToken> predPath = SentenceUtils.ancestorPath(predicate, ancestor);

		argPath.removeLast(); //ancestor is last thing in both pathA and pathB, don't need it's deprel
		predPath.removeLast();
		
		while(!argPath.isEmpty())	//argPath is (upward) path from argument to ancestor
			path.append(argPath.removeFirst().deprel + "^ ");
		while(!predPath.isEmpty())	//predPath is (downwards) path from predicate to ancestor
			path.append(predPath.removeLast().deprel + "v ");
		
		features.add("path|" + path.toString());
		features.add("pathpos|" + argument.pposs + " " + path.toString() + predicate.pposs);	//with pposs tags
		features.add("pathlem|" + argument.splitLemma + " " + path.toString() + predicate.splitLemma);	//with splm tagss

		
		/*
		 * Feature 4: length of dependency path
		 */
		features.add("pathlength|" + SentenceUtils.dependencyPathLength(predicate, argument));
		
		/*
		 * Feature 5: difference in positions, and binary tokens
		 */
		int distance = Math.abs(predicate.sentenceIndex - argument.sentenceIndex);
		features.add("distance|" + distance);
		features.add("distance=1|" + (distance==1?"true":"false"));
		features.add("distance=2|" + (distance==2?"true":"false"));
		features.add("distance>2|" + (distance>2?"true":"false"));
		
		/*
		 * Feature 6: predicate before or after argument
		 */
		features.add("predrelpos|" + ((predicate.sentenceIndex < argument.sentenceIndex) ? "before" : "after"));
		
		return features;
	}
	
	public static Dataset<String, String> dataSetFromCorpus(String corpusLoc) throws IOException{
		
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<List<ArgumentClassifierToken>> sentences = sentencesFromCorpus(corpusLoc);
		
		while (!sentences.isEmpty()){
			
			List<ArgumentClassifierToken> sentence = sentences.remove(0);
			List<ArgumentClassifierToken> predicates = new ArrayList<ArgumentClassifierToken>();
			for (ArgumentClassifierToken token : sentence){
				if (token.goldIsPredicate())
					predicates.add(token);
			}
			
			for (ArgumentClassifierToken predicate : predicates){
				for (ArgumentClassifierToken possibleArg : argumentCandidates(predicate))
				{
					String label = possibleArg.goldPredicateLabel(predicate);
					dataset.add(new BasicDatum<String, String>(
							getFeatures(predicate, possibleArg),
							label));
				}
			}
			
		}
		
		return dataset;
	}
	
	public String argClass(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return classifier.classOf(new BasicDatum<String, String>(getFeatures(predicate, argument)));
	}
	
	public Counter<String> argClassScores(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return classifier.scoresOf(new BasicDatum<String, String>(getFeatures(predicate, argument)));
	}
	
	public Map<ArgumentClassifierToken, String> argumentsOf(ArgumentClassifierToken predicate){
		
		Map<ArgumentClassifierToken, String> argumentLinks = new LinkedHashMap<ArgumentClassifierToken, String>();
		
		Map<ArgumentClassifierToken, Counter<String>> argumentLabelProbabilities =
				new HashMap<ArgumentClassifierToken, Counter<String>>();
		
		for(ArgumentClassifierToken possibleArg : ArgumentClassifier.argumentCandidates(predicate)){ //weed out NILs, SUs, AM-s
			Counter<String> argClassScores = argClassScores(possibleArg, predicate);
			String argMax = Counters.argmax(argClassScores);
			if (argMax.equals("NIL") || argMax.equals("SU") || argMax.startsWith("AM-"))
				argumentLinks.put(possibleArg, argMax);
			else{
				argumentLabelProbabilities.put(possibleArg, argClassScores);
			}
		}
		
		while (!argumentLabelProbabilities.isEmpty()){
			Pair<ArgumentClassifierToken, String> argAndLabel = mostCertainArgLabel(argumentLabelProbabilities);
			ArgumentClassifierToken registeredArg = argAndLabel.first();
			String argLabel = argAndLabel.second();
			argumentLabelProbabilities.remove(registeredArg);
			argumentLinks.put(registeredArg, argLabel);
			
			if (!argLabel.equals("NIL") && !argLabel.equals("SU") && !argLabel.startsWith("AM-")){
				
				Collection<Integer> restrictedIndices = registeredArg.getDescendantIndices();
				restrictedIndices.addAll(registeredArg.getAncestorIndices());
				
				for (ArgumentClassifierToken token : argumentLabelProbabilities.keySet()){
					if (restrictedIndices.contains(token.sentenceIndex)){
						Counter<String> tokenLabelProbabilities = argumentLabelProbabilities.get(token);
						Counter<String> updatedTokenLabelProbabilities = new ClassicCounter<String>();
						for (Map.Entry<String, Double> e : tokenLabelProbabilities.entrySet()){
							String label = e.getKey();
							if (label.equals("NIL") || label.equals("SU") || label.startsWith("AM-"))
								updatedTokenLabelProbabilities.setCount(label, e.getValue());
						}
						argumentLabelProbabilities.put(token, updatedTokenLabelProbabilities);
					}
				}
			}
			
			if (argLabel.matches("A[0-9]"))
				for (ArgumentClassifierToken token : argumentLabelProbabilities.keySet()){
					argumentLabelProbabilities.get(token).remove(argLabel);
				}
			
		}
		
		return argumentLinks;
	}
	
	private Pair<ArgumentClassifierToken, String> mostCertainArgLabel(Map<ArgumentClassifierToken, Counter<String>> argumentLabelProbabilities){
		Pair<ArgumentClassifierToken, String> mostCertainArgLabel = null;
		double highCertainty = Double.NEGATIVE_INFINITY;
		for (Entry<ArgumentClassifierToken, Counter<String>> token : argumentLabelProbabilities.entrySet()){
			String label = Counters.argmax(token.getValue());
			double certainty = token.getValue().getCount(label);
			if (certainty > highCertainty){
				highCertainty = certainty;
				mostCertainArgLabel = new Pair<ArgumentClassifierToken, String>(token.getKey(), label);
			}
		}
		
		return mostCertainArgLabel;
		
	}
	
}
