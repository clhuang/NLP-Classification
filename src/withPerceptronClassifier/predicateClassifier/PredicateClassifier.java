package withPerceptronClassifier.predicateClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


import util.CorpusUtils;
import withPerceptronClassifier.classify.PerceptronClassifier;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;

public class PredicateClassifier {
	
	private PerceptronClassifier classifier;
	
	public PredicateClassifier(PerceptronClassifier classifier){
		this.classifier = classifier;
	}
	
	public boolean isPredicate(Datum<String, String> d){
		return classifier.classOf(d).equals("predicate");
	}
	
	public boolean isPredicate(FeaturedPredicateToken t){
		return isPredicate(t.asDatum());
	}
	
	public List<? extends FeaturedPredicateToken> predicatesInSentence(List<? extends FeaturedPredicateToken> sentenceTokens){
		List<FeaturedPredicateToken> predicates = new ArrayList<FeaturedPredicateToken>();
		for (FeaturedPredicateToken t : sentenceTokens)
			if (isPredicate(t.asDatum()))
				predicates.add(t);
		return predicates;
	}
	
	public static List<? extends FeaturedPredicateToken> goldPredicatesInSentence(List<? extends FeaturedPredicateToken> sentenceTokens){
		List<FeaturedPredicateToken> predicates = new ArrayList<FeaturedPredicateToken>();
		for (FeaturedPredicateToken t : sentenceTokens)
			if (t.goldIsPredicate())
				predicates.add(t);
		return predicates;
	}
	
	public static Dataset<String, String> dataSetFromCorpus(String corpusLoc)
			throws NumberFormatException, IOException{

		List<List<String[]>> sentences = CorpusUtils.sentenceDataFromCorpus(corpusLoc);
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<FeaturedPredicateToken> sentenceTokens = new ArrayList<FeaturedPredicateToken>();

		System.out.println("Generating dataset");
		while(!sentences.isEmpty()){
			List<String[]> sentence = sentences.remove(0);
			for (String[] tokenData : sentence){
				sentenceTokens.add(new FeaturedPredicateToken(	//make new token, add to list
						tokenData[CorpusUtils.SPLIT_FORM_COLUMN],	//split_form
						tokenData[CorpusUtils.SPLIT_LEMMA_COLUMN],	//split_lemma
						tokenData[CorpusUtils.PPOSS_COLUMN],		//pposs
						tokenData[CorpusUtils.DEPREL_COLUMN], 		//deprel
						tokenData[CorpusUtils.PREDICATE_COLUMN],	//predicate role

						Integer.parseInt(tokenData[CorpusUtils.PARENT_INDEX_COLUMN]) - 1,	//parent index
						Integer.parseInt(tokenData[CorpusUtils.INDEX_COLUMN]) - 1, //this index
						//offset by 1 because array indices are 0-based, not 1-based

						sentenceTokens));		//list of sentence tokens	
			}
			
			for (FeaturedPredicateToken t : sentenceTokens){ 
				if (t.parentIndex >= 0){	//link parents to children
					FeaturedPredicateToken parent = sentenceTokens.get(t.parentIndex);
					parent.addChild(t.sentenceIndex);
				}
				t.updateAdjacentTokens();
			}

			for (FeaturedPredicateToken t : sentenceTokens){	//add tokens to dataset
				dataset.add(t.asDatum());
			}
			
			sentenceTokens.clear();
			
		}

		return dataset;

	}
	
	public static List<List<FeaturedPredicateToken>> sentencesFromCorpus(String corpusLoc) throws IOException {

		List<List<String[]>> sentenceData = CorpusUtils.sentenceDataFromCorpus(corpusLoc);
		List<List<FeaturedPredicateToken>> sentences = new ArrayList<List<FeaturedPredicateToken>>();
		List<FeaturedPredicateToken> sentenceTokens;

		while(!sentenceData.isEmpty()){
			List<String[]> sentence = sentenceData.remove(0);
			sentenceTokens = new ArrayList<FeaturedPredicateToken>();
			for (String[] tokenData : sentence){
				sentenceTokens.add(new FeaturedPredicateToken(	//make new token, add to list
						tokenData[CorpusUtils.SPLIT_FORM_COLUMN],	//split_form
						tokenData[CorpusUtils.SPLIT_LEMMA_COLUMN],	//split_lemma
						tokenData[CorpusUtils.PPOSS_COLUMN],		//pposs
						tokenData[CorpusUtils.DEPREL_COLUMN], 		//deprel
						tokenData[CorpusUtils.PREDICATE_COLUMN],	//predicate role

						Integer.parseInt(tokenData[CorpusUtils.PARENT_INDEX_COLUMN]) - 1,	//parent index
						Integer.parseInt(tokenData[CorpusUtils.INDEX_COLUMN]) - 1, //this index
						//offset by 1 because array indices are 0-based, not 1-based

						sentenceTokens));		//list of sentence tokens	
			}
			
			for (FeaturedPredicateToken t : sentenceTokens){ 
				if (t.parentIndex >= 0){	//link parents to children
					FeaturedPredicateToken parent = sentenceTokens.get(t.parentIndex);
					parent.addChild(t.sentenceIndex);
				}
				t.updateAdjacentTokens();
			}
			
			sentences.add(sentenceTokens);
			
		}

		return sentences;

	}
	
}
