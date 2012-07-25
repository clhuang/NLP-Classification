package predicateClassification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import util.CorpusUtils;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.ling.Datum;

public class PredicateClassifier {
	
	private LinearClassifier<String, String> linearClassifier;
	
	public PredicateClassifier(LinearClassifier<String, String> l){
		linearClassifier = l;
	}
	
	public boolean isPredicate(Datum<String, String> d){
		return linearClassifier.classOf(d).equals("predicate");
	}
	
	public static Dataset<String, String> dataSetFromCorpus(String corpusLoc)
			throws NumberFormatException, IOException{

		List<List<String[]>> sentences = CorpusUtils.sentencesFromCorpus(corpusLoc);
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<FeaturedPredicateToken> sentenceTokens = new ArrayList<FeaturedPredicateToken>();

		for(List<String[]> sentence : sentences){
			 
			for(String[] tokenData : sentence){
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
			
			for(FeaturedPredicateToken t : sentenceTokens){ //link parents to children
				if(t.parentIndex >= 0){
					FeaturedPredicateToken parent = sentenceTokens.get(t.parentIndex);
					parent.addChild(t.sentenceIndex);
				}
			}

			for(FeaturedPredicateToken t : sentenceTokens){	//add tokens to dataset
				dataset.add(t.asDatum());
			}
			
			sentenceTokens.clear();
			
		}

		return dataset;

	}
	
	
	
}
