package util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.util.ErasureUtils;

public class CorpusUtils {
	
	public static final int INDEX_COLUMN = 0;
	public static final int SPLIT_FORM_COLUMN = 5;
	public static final int SPLIT_LEMMA_COLUMN = 6;
	public static final int PPOSS_COLUMN = 7;
	public static final int PARENT_INDEX_COLUMN = 8;
	public static final int DEPREL_COLUMN = 9;
	public static final int PREDICATE_COLUMN = 10;
	public static final int ARGS_START_COLUMN = 11;
	
	public static List<List<String[]>> sentenceDataFromCorpus(String corpusLoc) throws IOException{
		List<List<String[]>> sentences = new ArrayList<List<String[]>>(); 
		
		String strLine;
		FileInputStream fstream = new FileInputStream(corpusLoc);
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));

		ArrayList<String[]> sentenceTokens = new ArrayList<String[]>();
		
		while ((strLine = br.readLine()) != null) {
			if (strLine.equals("")){	//sentence is over, enter sentence data into dataset
				List<String[]> sentenceTokensClone = ErasureUtils.uncheckedCast(sentenceTokens.clone());
				sentences.add(sentenceTokensClone);
				sentenceTokens.clear();
			}

			else{ 				//is a token in the same sentence
				sentenceTokens.add(strLine.split("\\s+"));
			}

		}
		
		return sentences;
	}
	
	
	/*
	 * In the training/development corpuses, tokens are represented as lines,
	 * with corresponding characteristics as corresponding "columns"
	 * in those lines.
	 * This function, given a line and a column number,
	 * returns the corresponding characteristic as a string. 
	 */
	public static String columnOf(String line, int num){
		String[] columns = line.split("\\s+");
		if (columns.length > num)
			return columns[num];
		return "";
	}
}
