package argumentClassification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;

import util.CorpusUtils;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.stats.Counter;

public class ArgumentClassifier {

	private LinearClassifier<String, String> linearClassifier;

	public ArgumentClassifier(LinearClassifier<String, String> linearClassifier){
		this.linearClassifier = linearClassifier;
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
	
	public static List<ArgumentClassifierToken> argumentCandidates(ArgumentClassifierToken predicate){
		List<ArgumentClassifierToken> candidates = new ArrayList<ArgumentClassifierToken>();
		
		for (ArgumentClassifierToken t : predicate.getSentenceTokens()){
			
			ArgumentClassifierToken ancestor = SentenceUtils.getCommonAncestor(t, predicate);
			int argumentAncestorPathLength = SentenceUtils.ancestorPathLength(t, ancestor);
			int predicateAncestorPathLength = SentenceUtils.ancestorPathLength(predicate, ancestor);
			
			if (argumentAncestorPathLength < 3 &&
					predicateAncestorPathLength < 5 &&
					argumentAncestorPathLength + predicateAncestorPathLength < 6)
				candidates.add(t);
			
		}
		
		return candidates;
	}
	
	public static List<List<ArgumentClassifierToken>> sentencesFromCorpus(String corpusLoc) throws IOException{
		List<List<String[]>> sentenceData = CorpusUtils.sentencesFromCorpus(corpusLoc);
		List<List<ArgumentClassifierToken>> sentences = new ArrayList<List<ArgumentClassifierToken>>();
		List<ArgumentClassifierToken> sentenceTokens;
		
		for (List<String[]> sentence : sentenceData){
			sentenceTokens = new ArrayList<ArgumentClassifierToken>();
			List<ArgumentClassifierToken> predicates = new ArrayList<ArgumentClassifierToken>();
			
			for (String[] tokenData : sentence){
				ArgumentClassifierToken token = new ArgumentClassifierToken(	//make new token, add to list
						tokenData[CorpusUtils.SPLIT_FORM_COLUMN],	//split_form
						tokenData[CorpusUtils.SPLIT_LEMMA_COLUMN],	//split_lemma
						tokenData[CorpusUtils.PPOSS_COLUMN],		//pposs
						tokenData[CorpusUtils.DEPREL_COLUMN], 		//deprel
						tokenData[CorpusUtils.PREDICATE_COLUMN],	//predicate role

						Integer.parseInt(tokenData[CorpusUtils.PARENT_INDEX_COLUMN]) - 1,	//parent index
						Integer.parseInt(tokenData[CorpusUtils.INDEX_COLUMN]) - 1, //this index
						//offset by 1 because array indices are 0-based, not 1-based

						sentenceTokens);		//list of sentence tokens
						
				sentenceTokens.add(token);
				if(token.isPredicate())
					predicates.add(token);
				
			}

			for (int i = 0; i < sentence.size(); i++){
				String[] tokenData = sentence.get(i);
				ArgumentClassifierToken token = sentenceTokens.get(i);
				for (int j = CorpusUtils.ARGS_START_COLUMN; j < tokenData.length; j++)	//link predicate arg to predicate
					if (!tokenData[j].equals("_")){
						int predicateNum = j - CorpusUtils.ARGS_START_COLUMN;
						token.addPredicate(predicates.get(predicateNum).sentenceIndex, tokenData[j]);
					}
			}
			
			
			for (ArgumentClassifierToken t : sentenceTokens){ //link parents to children
				if (t.parentIndex >= 0){
					sentenceTokens.get(t.parentIndex).addChild(t.sentenceIndex);
				}
			}
			
			sentences.add(sentenceTokens);

		}
		
		return sentences;
		
	}

	public static Dataset<String, String> dataSetFromCorpus(String corpusLoc) throws IOException{
		
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<List<ArgumentClassifierToken>> sentences = sentencesFromCorpus(corpusLoc);
		
		while (!sentences.isEmpty()){
			
			List<ArgumentClassifierToken> sentence = sentences.remove(0);
			List<ArgumentClassifierToken> predicates = new ArrayList<ArgumentClassifierToken>();
			for (ArgumentClassifierToken token : sentence){
				if (token.isPredicate())
					predicates.add(token);
			}
			
			for (ArgumentClassifierToken predicate : predicates){
				for (ArgumentClassifierToken possibleArg : argumentCandidates(predicate))
				{
					dataset.add(new BasicDatum<String, String>(
							getFeatures(predicate, possibleArg),
							possibleArg.predicateLabel(predicate)));
				}
			}
			
		}
		
		return dataset;
	}
	
	public String argClass(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return linearClassifier.classOf(new BasicDatum<String, String>(getFeatures(predicate, argument)));
	}
	
	public Double probabilityIsArgument(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return 1 - argClassProbabilities(argument, predicate).getCount("NIL");
	}
	
	public Counter<String> argClassProbabilities(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return linearClassifier.probabilityOf(new BasicDatum<String, String>(getFeatures(predicate, argument)));
	}
	
	/*public List<ArgumentWithProbability> sortedPossibleArgs(ArgumentClassifierToken predicate){
		List<ArgumentWithProbability> args = new ArrayList<ArgumentWithProbability>();
		for (ArgumentClassifierToken t : argumentCandidates(predicate)){
			args.add(new ArgumentWithProbability(t, probabilityIsArgument(t, predicate)));
		}
		Collections.sort(args);
		return args;
	}
	
	public class ArgumentWithProbability implements Comparable<ArgumentWithProbability>{
		private ArgumentClassifierToken argument;
		public final Double probability;
		private ArgumentWithProbability(ArgumentClassifierToken possibleArgument, Double probability){
			this.argument = possibleArgument;
			this.probability = probability;
		}
		
		public int compareTo(ArgumentWithProbability o) {
			return o.probability.compareTo(this.probability);
		}
		
		public ArgumentClassifierToken asToken(){
			return argument;
		}
	}*/

}
