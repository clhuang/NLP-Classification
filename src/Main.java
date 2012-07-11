
import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Main {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		String strLine;
		FileInputStream fstream = new FileInputStream("test.closed");
		DataInputStream in = new DataInputStream(fstream);
		BufferedReader br = new BufferedReader(new InputStreamReader(in));

		LinearClassifierFactory<String, String> linearClassifierFactory =
				new LinearClassifierFactory<String, String>();
		Dataset<String, String> dataset = new Dataset<String, String>();
		List<FeaturedToken> sentenceFeaturedTokens = new ArrayList<FeaturedToken>();

		while ((strLine = br.readLine()) != null) {
			if(strLine.equals("")){	//sentence is over, enter sentence data into dataset
				for(FeaturedToken t : sentenceFeaturedTokens){ //link parents to children
					if(t.parentIndex >= 0){
						FeaturedToken parent = sentenceFeaturedTokens.get(t.parentIndex);
						parent.addChild(t.sentenceIndex);
					}
				}

				for(FeaturedToken t : sentenceFeaturedTokens){
					Collection<String> features;
					features = t.computeFeatures();

					for(String s : features)
						System.out.println(s);

					String label = t.isPredicate ? "yes" : "no";	//represents if a predicate
					Datum<String, String> d = new BasicDatum<String, String>(features, label);
					dataset.add(d);
				}
				System.out.println();
				System.out.println();
				sentenceFeaturedTokens.clear(); //begin new sentence

			}

			else{ 				//is a token
				sentenceFeaturedTokens.add(new FeaturedToken(
						columnOf(strLine, 5),					//split_form
						columnOf(strLine, 6),			 		//split_lemma
						columnOf(strLine, 7),					//pposs
						columnOf(strLine, 9), 					//deprel
						columnOf(strLine, 10).equals("_"),		//is predicate

						Integer.parseInt(columnOf(strLine, 8)) - 1,	//parent index
						Integer.parseInt(columnOf(strLine, 0)) - 1, //this index
						//offset by 1 because array indices are 0-based, not 1-based

						sentenceFeaturedTokens));		//list of sentence tokens	
			}

		}

		LinearClassifier<String, String> linearClassifier =
				linearClassifierFactory.trainClassifier(dataset);
	}

	private static String columnOf(String s, int num){
		String[] columns = s.split("\\s+");
		if (columns.length > num)
			return columns[num];
		return "";
	}

}
