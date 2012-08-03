package argumentClassification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.stats.Counter;

import util.CorpusUtils;

public abstract class ArgumentClassifier {
	
	protected LinearClassifier<String, String> linearClassifier;
	
	public ArgumentClassifier(LinearClassifier<String, String> linearClassifier){
		this.linearClassifier = linearClassifier;
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
	
	protected static List<String> sortArgLabels(Counter<String> argCounter) {
		List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(argCounter.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
	               return (o2.getValue())
	              .compareTo(o1.getValue());
	          }
		});
		List<String> sortedLabels = new ArrayList<String>();
		for (Map.Entry<String, Double> entry : list)
			sortedLabels.add(entry.getKey());
		return sortedLabels;
	}
	
	public abstract Map<ArgumentClassifierToken, String> argumentsOf(ArgumentClassifierToken predicate);
	
	public static Map<ArgumentClassifierToken, String> goldArgumentsOf(ArgumentClassifierToken predicate){
		Map<ArgumentClassifierToken, String> argClasses = new LinkedHashMap<ArgumentClassifierToken, String>();
		for (ArgumentClassifierToken argument : ArgumentClassifier.argumentCandidates(predicate)){
			argClasses.put(argument, argument.predicateLabel(predicate));
		}
		return argClasses;
	}
	
}
