
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		LinearClassifierFactory<String, String> linearClassifierFactory = new LinearClassifierFactory<String, String>();
		Dataset<String, String> dataset = new Dataset<String, String>();
		
		Collection sentences = null;							//TODO get sentences
		
		for(Object o : sentences){
			List<FeaturedToken> sentenceFeaturedTokens = new ArrayList<FeaturedToken>();//TODO get sentence FeaturedTokens
			for(FeaturedToken t : sentenceFeaturedTokens){
				Collection<String> features;
				features = t.computeFeatures();
				
				String label = t.isPredicate ? "yes" : "no";//label, represents whether or not is a predicate
				Datum<String, String> d = new BasicDatum<String, String>(features, label);
				dataset.add(d);
			}
		}
		
		LinearClassifier<String, String> linearClassifier = linearClassifierFactory.trainClassifier(dataset);
		
	}

}
