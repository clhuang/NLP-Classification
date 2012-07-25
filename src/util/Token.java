package util;

import java.util.Collections;
import java.util.List;

public class Token {
	
	public final String splitForm;
	public final String splitLemma;
	public final String pposs;
	public final String deprel;
	public final String predicateRole;
	
	public final int parentIndex;
	public final int sentenceIndex;
	
	protected List<? extends Token> sentenceTokens;
	
	public Token(String splitForm, String splitLemma,
			String pposs, String deprel, String predicateRole,
			int parentIndex, int sentenceIndex, List<? extends Token> sentenceTokens){
		
		this.splitForm = splitForm;
		this.splitLemma = splitLemma;
		this.pposs = pposs;
		this.deprel = deprel;
		this.parentIndex = parentIndex;
		this.predicateRole = predicateRole;
		this.sentenceIndex = sentenceIndex;
		this.sentenceTokens = sentenceTokens;
		
	}
	
	public List<? extends Token> getSentenceTokens(){
		return Collections.unmodifiableList(sentenceTokens);
	}
	
	public Token getParent() {
		if(parentIndex >= 0)
			return sentenceTokens.get(parentIndex);
		return null;
	}
	
}

