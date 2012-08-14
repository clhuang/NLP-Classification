package withPerceptronClassifier.classify;

import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;

public class PerceptronClassifier implements Serializable{

	private static final long serialVersionUID = 1L;

	static class LabelWeights implements Serializable{

		private static final long serialVersionUID = 1L;
		double[] weights;
		int survivalIterations;
		double[] avgWeights;

		LabelWeights(int numFeatures) {
			weights = new double[numFeatures];
			avgWeights = new double[numFeatures];
			survivalIterations = 0;
		}

		void clear() {
			weights = null;
		}

		void incrementSurvivalIterations() {
			survivalIterations++;
		}

		private void updateAverage() {
			for(int i = 0; i < weights.length; i++){
				avgWeights[i] += weights[i] * survivalIterations;
			}
		}

		void update(Set<Integer> exampleFeatureIndices, double weight) {
			updateAverage();

			for(int d : exampleFeatureIndices){
				weights[d] += weight;
			}

			survivalIterations = 0;
		}
		
		void normalize(double norm) {
			if(norm > 0)
				for(int i = 0; i < avgWeights.length; i++)
					avgWeights[i] /= norm;
		}
		
		double dotProduct(Set<Integer> featureIndices) {
			return dotProduct(featureIndices, weights);
		}
		
		static double dotProduct(Set<Integer> featureCounts, double [] weights) {
			double dotProd = 0;
			for (int i : featureCounts)
				dotProd += weights[i];
			return dotProd;
		}
		
	}
	
	LabelWeights[] zWeights;
	
	public Index<String> labelIndex;
	public Index<String> featureIndex;
	
	final int epochs;

	public PerceptronClassifier(int epochs){
		this.epochs = epochs;
	}
	
	public PerceptronClassifier(){
		this(10);
	}
	
	public void save(String modelPath) throws IOException {
		/*for(LabelWeights zw : zWeights){
			zw.clear();
		}*/
		
		ObjectOutputStream out = new ObjectOutputStream(new BufferedOutputStream(
		        new GZIPOutputStream(new FileOutputStream(modelPath))));
		
		assert(zWeights != null);
		out.writeInt(zWeights.length);
	    for(LabelWeights zw: zWeights) {
	      out.writeObject(zw);
	    }
	    
	    out.writeObject(labelIndex);
	    out.writeObject(featureIndex);
	    
	    out.close(); 
	}

	public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
		int length = in.readInt();
		zWeights = new LabelWeights[length];
		for(int i = 0; i < zWeights.length; i ++){
			zWeights[i] = ErasureUtils.uncheckedCast(in.readObject());
		}

		labelIndex = ErasureUtils.uncheckedCast(in.readObject());
		featureIndex = ErasureUtils.uncheckedCast(in.readObject());
	}
	
	public static PerceptronClassifier load(String modelPath) throws IOException, ClassNotFoundException {
		GZIPInputStream is = new GZIPInputStream(new FileInputStream(modelPath));
	
		ObjectInputStream in = new ObjectInputStream(is);
		PerceptronClassifier ex = new PerceptronClassifier();
		ex.load(in);
		in.close();
		is.close();
		return ex;
	}
	
	public void train(Dataset<String, String> dataset){
		labelIndex = dataset.labelIndex();
		featureIndex = dataset.featureIndex();
		int numFeatures = featureIndex.size();
		
		zWeights = new LabelWeights[labelIndex.size()];
		for(int i = 0; i < zWeights.length; i++)
			zWeights[i] = new LabelWeights(numFeatures);
		
		System.err.println("Running perceptronClassifier on " + dataset.size() + " features");
		long startTime = System.currentTimeMillis();
		
		for(int t = 0; t < epochs; t++){
			dataset.randomize(t);
			
			System.err.println();
			System.err.println("Epoch: " + (t+1) + " of " + epochs);
			
			for(int i = 0; i < dataset.size(); i++){
				if (i%500000 == 0){
					System.err.println("Datum: " + i + " of " + dataset.size());
					System.err.println("Elapsed time: " + (System.currentTimeMillis() - startTime)/1000 + "s");
				}
				Datum<String, String> datum = dataset.getDatum(i);
				
				Set<Integer> exampleFeatureIndices = featuresOf(datum);
				
				String predictedArg = argMaxDotProduct(exampleFeatureIndices);
				String goldArg = datum.label();
				if (!predictedArg.equals(goldArg)){
					zWeights[labelIndex.indexOf(predictedArg)].update(exampleFeatureIndices, -1.0);
					zWeights[labelIndex.indexOf(goldArg)].update(exampleFeatureIndices, 1.0);
				}
				
				for(LabelWeights zw : zWeights) {
					zw.incrementSurvivalIterations();
				}
			}
			
			int correct = 0;
			int encountered = 0;
			
			for(int i = 0; i < dataset.size(); i++){
				Datum<String, String> datum = dataset.getDatum(i);
				
				String predictedArg = trainingClassOf(datum);
				String goldArg = datum.label();
				if (predictedArg.equals(goldArg))
					correct++;
				encountered++;
			}
			System.out.println("Number correct: " + correct + "/" + encountered);
		}
	}
	
	private Set<Integer> featuresOf(Datum<String, String> datum){
		Set<Integer> featureIndices = new HashSet<Integer>();
		for(String feature : datum.asFeatures()){
			int index = featureIndex.indexOf(feature);
			if (index > 0)
				featureIndices.add(index);
		}
		return featureIndices;
	}
	
	private String argMaxDotProduct(Set<Integer> exampleFeatureIndices){
		double maxDotProduct = Double.NEGATIVE_INFINITY;
		int argMax = -1;
		for(int i = 0; i < zWeights.length; i++){
			double dotProduct = zWeights[i].dotProduct(exampleFeatureIndices);
			if (dotProduct > maxDotProduct){
				maxDotProduct = dotProduct;
				argMax = i;
			}
		}
		return labelIndex.get(argMax);
	}
	
	private String argMaxAverageDotProduct(Set<Integer> exampleFeatureIndices){
		double maxDotProduct = Double.NEGATIVE_INFINITY;
		int argMax = -1;
		for(int i = 0; i < zWeights.length; i++){
			double dotProduct = LabelWeights.dotProduct(exampleFeatureIndices, zWeights[i].avgWeights);
			if (dotProduct > maxDotProduct){
				maxDotProduct = dotProduct;
				argMax = i;
			}
		}
		return labelIndex.get(argMax);
	}
	
	public Counter<String> scoresOf(Datum<String, String> datum){
		Counter<String> scores = new ClassicCounter<String>();
		Set<Integer> featureCounts = featuresOf(datum);
		for (int i = 0; i < labelIndex.size(); i++){
			scores.incrementCount(labelIndex.get(i),
					LabelWeights.dotProduct(featureCounts, zWeights[i].avgWeights));
		}
		return scores;
	}
	
	public String trainingClassOf(Datum<String, String> datum){
		return argMaxDotProduct(featuresOf(datum));
	}
	
	public String classOf(Datum<String, String> datum){
		return argMaxAverageDotProduct(featuresOf(datum));
	}

}
