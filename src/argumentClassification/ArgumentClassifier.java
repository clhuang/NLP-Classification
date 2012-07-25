package argumentClassification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.List;
import util.CorpusUtils;
import util.Token;
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.BasicDatum;

public class ArgumentClassifier {

	private LinearClassifier<String, String> linearClassifier;

	public ArgumentClassifier(LinearClassifier<String, String> linearClassifier){
		this.linearClassifier = linearClassifier;
	}

	public Collection<String> getFeatures(ArgumentClassifierToken predicate,
			ArgumentClassifierToken argument){
		
		if (argument.getSentenceTokens() != predicate.getSentenceTokens())
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
		if(pmod != null){
			features.add("pmodsplm|" + pmod.splitLemma);
			features.add("pmodspfm|" + pmod.splitForm);
			features.add("pmodppos|" + pmod.pposs);
		}
		
		//TODO: 2
		
		/*
		 * Feature 3: dependency path
		 */
		StringBuilder path = new StringBuilder();
		
		/*
		 * Ancestor splits the path into two halves: 
		 * from the argument to the ancestor, the dependencies go upwards;
		 * from the predicate to the ancestor, the dependencies go downwards
		 */
		Token ancestor = SentenceUtils.getCommonAncestor(argument, predicate);
		
		Deque<Token> argPath = SentenceUtils.ancestorPath(argument, ancestor);
		Deque<Token> predPath = SentenceUtils.ancestorPath(predicate, ancestor);

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
	
	private List<ArgumentClassifierToken> argumentCandidates(ArgumentClassifierToken predicate){
		List<ArgumentClassifierToken> candidates = new ArrayList<ArgumentClassifierToken>();
		
		for(ArgumentClassifierToken t : predicate.getSentenceTokens()){
			if (t.equals(predicate))
				continue;
			
			ArgumentClassifierToken ancestor = (ArgumentClassifierToken) SentenceUtils.getCommonAncestor(t, predicate);
			int argumentAncestorPathLength = SentenceUtils.ancestorPathLength(t, ancestor);
			int predicateAncestorPathLength = SentenceUtils.ancestorPathLength(predicate, ancestor);
			
			if(argumentAncestorPathLength < 3 &&
					predicateAncestorPathLength < 5 &&
					argumentAncestorPathLength + predicateAncestorPathLength < 6)
				candidates.add(t);
			
		}
		
		return candidates;
	}

	public Dataset<String, String> dataSetFromCorpus(String corpusLoc) throws IOException{
		List<List<String[]>> sentences = CorpusUtils.sentencesFromCorpus(corpusLoc);
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<ArgumentClassifierToken> sentenceTokens = new ArrayList<ArgumentClassifierToken>();

		for(List<String[]> sentence : sentences){
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
				if (token.isPredicate())
					predicates.add(token);
				
				for (int i = CorpusUtils.ARGS_START_COLUMN; i < tokenData.length; i++)	//link predicate arg to predicate
					if (tokenData[i] != "_"){
						int predicateNum = i - CorpusUtils.ARGS_START_COLUMN;
						token.addPredicate(predicates.get(predicateNum), tokenData[i]);
					}
			}

			for (ArgumentClassifierToken t : sentenceTokens){ //link parents to children
				if(t.parentIndex >= 0){
					ArgumentClassifierToken parent = sentenceTokens.get(t.parentIndex);
					parent.addChild(t.sentenceIndex);
				}
			}
			
			for (ArgumentClassifierToken predicate : predicates){
				for (ArgumentClassifierToken possibleArg : argumentCandidates(predicate))
				{
					Collection<String> features = getFeatures(predicate, possibleArg);
					String label = possibleArg.predicateLabel(predicate);
					dataset.add(new BasicDatum<String, String>(features, label));
				}
			}

		}

		return dataset;
	}
	
	public String argClass(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return linearClassifier.classOf(new BasicDatum<String, String>(getFeatures(predicate, argument)));
	}
	
	public boolean isArgument(ArgumentClassifierToken argument, ArgumentClassifierToken predicate){
		return linearClassifier.classOf(new BasicDatum<String, String>(getFeatures(predicate, argument))).equals("NIL");
	}

}
