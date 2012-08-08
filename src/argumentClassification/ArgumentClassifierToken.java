package argumentClassification;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import predicateClassification.FeaturedPredicateToken;

public class ArgumentClassifierToken extends FeaturedPredicateToken{
	
	private Map<Integer, String> labels = new HashMap<Integer, String>();

	public ArgumentClassifierToken(String splitForm, String splitLemma,
			String pposs, String deprel, String predicateRole, int parentIndex,
			int sentenceIndex, List<ArgumentClassifierToken> sentenceTokens) {
		super(splitForm, splitLemma, pposs, deprel, predicateRole, parentIndex,
				sentenceIndex, sentenceTokens);
	}
	
	public void addChild(int childIndex) {
		if (childIndex >= 0)
			childrenIndices.add(childIndex);
	}
	
	public boolean isPredicate() {
		return !(predicateRole.equals("_"));
	}

	public ArgumentClassifierToken getPMOD() {
		for (Integer i : childrenIndices)
			if (sentenceTokens.get(i).deprel.startsWith("PMOD"))
				return (ArgumentClassifierToken) sentenceTokens.get(i);
		return null;
	}
	
	public void addPredicate(int predicateIndex, String label){
		labels.put(predicateIndex, label);
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
	
	public List<ArgumentClassifierToken> getChildren(){
		List<ArgumentClassifierToken> children = new ArrayList<ArgumentClassifierToken>();
		for (Integer index : childrenIndices)
			children.add((ArgumentClassifierToken) sentenceTokens.get(index));
		
		return children;
	}
	
	public Collection<Integer> getDescendantIndices(){
		Collection<Integer> descendantIndices = new HashSet<Integer>();
		for (Integer i : childrenIndices){
			ArgumentClassifierToken child = (ArgumentClassifierToken) sentenceTokens.get(i);
			descendantIndices.add(i);
			descendantIndices.addAll(child.getDescendantIndices());
		}
		return descendantIndices;
	}
	
	public Collection<Integer> getAncestorIndices(){
		Collection<Integer> ancestorIndices = new HashSet<Integer>();
		ArgumentClassifierToken ancestor = this;
		while (ancestor.parentIndex >= 0){
			ancestorIndices.add(ancestor.parentIndex);
			ancestor = (ArgumentClassifierToken) ancestor.getParent();
		}
		return ancestorIndices;
	}
	
	
	
}
