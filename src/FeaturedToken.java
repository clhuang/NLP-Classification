import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import edu.stanford.nlp.process.WordShapeClassifier;

public class FeaturedToken{
	public final String word;
	public final String pos;
	public final String entity;
	public final String deprel;

	public final boolean isPredicate;
	private List<Integer> childrenIndices;
	public final Integer numChildren;
	private List<FeaturedToken> sentenceTokens;
	public final int sentenceIndex;

	private final FeaturedToken prev2Token;
	private final FeaturedToken prevToken;
	private final FeaturedToken nextToken;
	private final FeaturedToken next2Token;
	
	public static final int wordShaper = WordShapeClassifier.WORDSHAPECHRIS2;

	FeaturedToken(final String word, String pos, String entity,
			String deprel, boolean isPredicate,
			List<Integer> childrenIndices,  List<FeaturedToken> sentenceTokens){
		
		this.word = word;
		this.isPredicate = isPredicate;
		this.pos = pos;
		this.entity = entity;
		this.deprel = deprel;

		this.sentenceTokens = sentenceTokens;
		sentenceIndex = sentenceTokens.indexOf(this);

		this.childrenIndices = childrenIndices;
		numChildren = childrenIndices.size();

		if(sentenceIndex > 0)
			prevToken = sentenceTokens.get(sentenceIndex - 1);
		else
			prevToken = null;

		if(sentenceIndex > 1)
			prev2Token = sentenceTokens.get(sentenceIndex - 2);
		else
			prev2Token = null;

		if(sentenceIndex < sentenceTokens.size() - 1)
			nextToken = sentenceTokens.get(sentenceIndex + 1);
		else
			nextToken = null;

		if(sentenceIndex < sentenceTokens.size() - 2)
			next2Token = sentenceTokens.get(sentenceIndex + 2);
		else
			next2Token = null;
	}
	
	/*
	 * Features are used by the linear classifier to classify tokens.
	 * Features are provided as strings in the following format:
	 * [feature identifier]|[data]
	 * 
	 * ex: number of children for a word with 11 children:
	 * numch|11
	 * 
	 * ex: word shape for "Bush":
	 * wdshp|Xxxx
	 * 
	 * List of prefixes:
	 * splm[u/b] -- split-lemma (unigram/bigram)
	 * spfm -- split-form (unigram)
	 * ppos[u/b] -- pposs (unigram/bigram)
	 * wdshp -- word shape (string)
	 * numch -- number of children (integer)
	 * c[#] -- child
	 * 
	 */
	public Collection<String> computeFeatures() {

		Collection<String> features = new HashSet<String>();

		features.addAll(getSplitLemma());		//add split-lemma
		features.addAll(getSplitForm());		//add split-form
		features.addAll(getPPoss());			//add pposs
		
		features.add("wdshp|" +					//add word shape using Stanford's WordShapeClassifier
				WordShapeClassifier.wordShape(word, wordShaper));
		
		features.add("numch|" +	numChildren);	//add number of children
		
		features.addAll(getChildrenFeatures()); //add children features
		features.add(getChildrenDifferences()); //add children differences

		return features;
	}


	/*
	 * Returns a list of differences between this token's number and its children's numbers in the format:
	 * chdif|[space-delimited numbers]
	 * 
	 * ex: getChildrenDifferences of a token (at position 17) with children at positions 3, 14, 22, 23
	 * chdif|-14 -3 5 6
	 * order of children differences not guaranteed
	 * 
	 */
	private String getChildrenDifferences() {
		StringBuilder s = new StringBuilder("chdif|");
		for(Integer i : childrenIndices)
			s.append(" " + (i - sentenceIndex));
		
		return s.toString();
	}

	/*
	 * Collection of children's features
	 * 
	 * Each string in the child's pposs/split-lemma is represented here:
	 * c[the child's index in sentence][string in child's feature]
	 * 
	 * ex: i feature of split lemma of child token "run" (number 17 in sentence)
	 * c17splmu,i|run|O|VB
	 * 
	 * 
	 * Child's relation (deprel) to parent is represented:
	 * c[child's sentence index]deprel|[deprel]
	 * 
	 * ex: deprel of child token (number 2) "president" is "APPO"
	 * c2deprel|APPO
	 * 
	 * 
	 * Child's pposs/split-lemmas are also added, concatenated with the parent's
	 * 
	 */
	private Collection<String> getChildrenFeatures(){
		Collection<String> childFeatures = new HashSet<String>();
		for(Integer i : childrenIndices){
			//add child split-lemma
			FeaturedToken child = sentenceTokens.get(i);
			for(String s : child.getSplitLemma())
				childFeatures.add("c" + child.sentenceIndex + s);
			
			//add child pposs
			for(String s : child.getPPoss())
				childFeatures.add("c" + child.sentenceIndex + s);
			
			//add child deprel
			childFeatures.add("c" + child.sentenceIndex + "deprel|" + child.deprel);
			
			//TODO add <split_lemma(this), split_lemma(child)>
			//how do we represent concatenation of features?
			
			//TODO add <pposs(this), pposs(child)>
			
		}
		
		return childFeatures;
	}

	/*
	 * Collection of pposs features in format:
	 * ppos[u/b],relative word position(s)|word(s)|entity(s)|pos(s)
	 * words, entities, pos's in bigrams are space-delimited
	 * 
	 * ex: i-1 token of "drives" in "George Bush drives"
	 * pposu,i-1|Bush|PERSON|NNP
	 * 
	 * ex: i+1,i+2 tokens of "George" in "George Bush drives
	 * pposb,i+1,i+2|Bush drives|PERSON O|NNP VBZ
	 * 
	 */
	public Collection<String> getPPoss() {
		Collection<String> pposs = new HashSet<String>();
		
		//pposs unigrams
		if(sentenceIndex > 0)
			pposs.add("pposu,i-1|"	+		//i-1
					prevToken.word + "|" +
					prevToken.entity + "|" +
					prevToken.pos);
		
		pposs.add("pposu,i|" +				//i
				word + "|" +
				entity + "|" +
				pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			pposs.add("pposu,i+1|"	+		//i+1
					nextToken.word + "|" +
					nextToken.entity + "|" +
					nextToken.pos);

		//pposs bigrams
		if(sentenceIndex > 1)
			pposs.add("pposb,i-2,i-1|" +	//<i-2, i-1>
					prev2Token.word + " " + prevToken.word + "|" + 
					prev2Token.entity + " " + prevToken.entity + "|" + 
					prev2Token.pos + " " + prevToken.pos);
		
		if(sentenceIndex > 0)
			pposs.add("pposb,i-1,i|" +		//<i-1, i>
					prevToken.word + " " + word + "|" + 
					prevToken.entity + " " + entity + "|" + 
					prevToken.pos + " " + pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			pposs.add("pposb,i,i+1|" +		//<i, i+1>
					word + " " + nextToken.word + "|" + 
					entity + " " + nextToken.entity + "|" + 
					pos + " " + nextToken.pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			pposs.add("pposb,i+1,i+2|" +	//<i, i+1>
					nextToken.word + " " + next2Token.word + "|" + 
					nextToken.entity + " " + next2Token.entity + "|" + 
					nextToken.pos + " " + next2Token.pos);
		
		return pposs;
		
	}

	/*
	 * Collection of split form features in format:
	 * spfm,relative word position|word|entity|pos
	 * 
	 * ex: i-1 token of "drives" in "George Bush drives"
	 * pposu,i-1|Bush|PERSON|NNP
	 * 
	 */
	public Collection<String> getSplitForm() {
		Collection<String> splitForm = new HashSet<String>();
		
		//split-form unigrams
		if(sentenceIndex > 1)
			splitForm.add("spfm,i-2|" +		//i-2
					prev2Token.word + "|" +
					prev2Token.entity + "|" + 
					prev2Token.pos);
		
		if(sentenceIndex > 0)
			splitForm.add("spfm,i-1|"	+		//i-1
					prevToken.word + "|" +
					prevToken.entity + "|" +
					prevToken.pos);
		
		splitForm.add("spfm,i|" +				//i
				word + "|" +
				entity + "|" +
				pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			splitForm.add("spfm,i+1|"	+		//i+1
					nextToken.word + "|" +
					nextToken.entity + "|" +
					nextToken.pos);
		
		if(sentenceIndex < sentenceTokens.size() - 2)
			splitForm.add("spfm,i+2|"	+		//i+2
					next2Token.word + "|" +
					next2Token.entity + "|" +
					next2Token.pos);
		
		return splitForm;

	}

	/*
	 * Collection of pposs features in format:
	 * splm[u/b],relative word position(s)|word(s)|entity(s)|pos(s)
	 * words, entities, pos's in bigrams are space-delimited
	 * 
	 * ex: i-1 token of "drives" in "George Bush drives"
	 * splmu,i-1|Bush|PERSON|NNP
	 * 
	 * ex: i,i+1 tokens of "Bush" in "George Bush drives
	 * splmb,i,i+1|Bush drives|PERSON O|NNP VBZ
	 * 
	 */
	public Collection<String> getSplitLemma() {
		Collection<String> splitLemma = new HashSet<String>();

		//split-lemma unigrams
		if(sentenceIndex > 0)
			splitLemma.add("splmu,i-1|"	+		//i-1
					prevToken.word + "|" +
					prevToken.entity + "|" +
					prevToken.pos);
		
		splitLemma.add("splmu,i|" +				//i
				word + "|" +
				entity + "|" +
				pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			splitLemma.add("splmu,i+1|"	+		//i+1
					nextToken.word + "|" +
					nextToken.entity + "|" +
					nextToken.pos);
		

		//split-lemma bigrams
		if(sentenceIndex > 0)
			splitLemma.add("splmb,i-1,i|" +		//<i-1, i>
					prevToken.word + " " + word + "|" + 
					prevToken.entity + " " + entity + "|" + 
					prevToken.pos + " " + pos);
		
		if(sentenceIndex < sentenceTokens.size() - 1)
			splitLemma.add("splmb,i,i+1|" +		//<i, i+1>
					word + " " + nextToken.word + "|" + 
					entity + " " + nextToken.entity + "|" + 
					pos + " " + nextToken.pos);
		
		return splitLemma;

	}

}