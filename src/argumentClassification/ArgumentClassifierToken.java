package argumentClassification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import util.Token;

public class ArgumentClassifierToken extends Token{
	private List<Integer> childrenIndices;
	private List<ArgumentClassifierToken> arguments = new ArrayList<ArgumentClassifierToken>();
	
	private Map<Integer, String> labels = new HashMap<Integer, String>();

	public ArgumentClassifierToken(String splitForm, String splitLemma,
			String pposs, String deprel, String predicateRole, int parentIndex,
			int sentenceIndex, List<ArgumentClassifierToken> sentenceTokens) {
		super(splitForm, splitLemma, pposs, deprel, predicateRole, parentIndex,
				sentenceIndex, sentenceTokens);
	}
	
	public void addChild(int childIndex) {
		if(childIndex >= 0)
			childrenIndices.add(childIndex);
	}
	
	public boolean isPredicate() {
		return predicateRole.equals("_");
	}
	
	public void addArgument(ArgumentClassifierToken t, int argNum){
		if (isPredicate())
			arguments.add(argNum, t);
	}

	public ArgumentClassifierToken getPMOD() {
		for(Integer i : childrenIndices)
			if(sentenceTokens.get(i).deprel.startsWith("PMOD"))
				return (ArgumentClassifierToken) sentenceTokens.get(i);
		return null;
	}
	
	public void addPredicate(ArgumentClassifierToken predicate, String label){
		if (predicate.getSentenceTokens() != getSentenceTokens())
			return;
		labels.put(predicate.sentenceIndex, label);
	}
	
	public String predicateLabel(ArgumentClassifierToken predicate){
		String label = labels.get(predicate.sentenceIndex);
		if (label == null)
			return "NIL";
		return label;
	}
	
	@SuppressWarnings("unchecked")
	public List<ArgumentClassifierToken> getSentenceTokens(){
		return (List<ArgumentClassifierToken>) super.getSentenceTokens();
	}
}
