package weka;

//Andrea Castillo
import java.io.BufferedReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Perceptron extends Classifier implements weka.core.OptionHandler {

	String fileName;
	int EPOCH = 0;
	double bias = 1.0, actual, predicted, learningRate;
	Instances data;
	boolean classified;
	double[] weights;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		this.data = data;
		double sum = 0;
		int x = 0;
		weights = new double[data.numAttributes()];
		weights[2] = .5;
		weights[1] = .75;
		weights[0] = -.6;
		System.out.println(data.numAttributes() + " " + data.numInstances());
		System.out.println(" Argruments  epoch learning rate " + EPOCH + " " + learningRate);
		for (int i = 0; i < EPOCH; i++) {
			for (int j = 0; j < data.numInstances(); j++) {
				for (x = 0; x < data.numAttributes(); x++) {
					// System.out.print(data.instance(j).value(x) + " ");
					if (x == 0)
						sum = (weights[0] * bias);
					else
						sum += (weights[x] * data.instance(j).value(x - 1));
				}
				// A is 0 B is 1 I am changing it to A is 1 and B is -1
				
				actual = data.instance(j).value(x-1) == 0.0 ? 1.0 : -1.0;
				predicted = sum >= 0.0 ? 1.0 : -1.0;

				if (predicted == actual) {
					classified=true;
					System.out.printf("1");
				} else {
					classified=false;
					System.out.printf("0");
					if (actual == 1.0 && predicted == -1.0) {
						for (int m = 0; m < weights.length; m++) {
							if (m == 0)
								weights[0] = weights[0] + (2 * learningRate * bias);
							else
								weights[m] = weights[m] + (2 * learningRate * data.instance(j).value(m - 1));
						}
					}
					if (actual == -1.0 && predicted == 1.0) {
						for (int m = 0; m < weights.length; m++) {
							if (m == 0)
								weights[0] = weights[0] + (-2 * learningRate * bias);
							else
								weights[m] = weights[m] + (-2 * learningRate * data.instance(j).value(m - 1));
						}
					}
				}
			}
			System.out.println();

		}
	}

	public double[] distributionForInstance(Instance instance) {
		double[] result = new double[2];
		if (classified) {
			result[0] = 1;
			result[1] = 0;
		} else {
			result[0] = 0;
			result[1] = 1;
		}
		return result;

	}

	public void setOptions(String[] options) throws IOException, Exception {
		BufferedReader datafile = SimpleWeka.readDataFile(options[1]);
		fileName = options[1];
		EPOCH = (Integer.parseInt(options[3]));
		learningRate = (Double.parseDouble(options[5]));
	}

	public String toString() {
		String output = "Final weights:\n";
		
		for(int i = 0; i < weights.length; i++){
			output += (Math.round(weights[i] * 100.0) / 100.0) + "\n";
		}
//		 weights[0] = (Math.round(weights[0] * 100.0) / 100.0);
//		 weights[1] = (Math.round(weights[1] * 100.0) / 100.0);
//		 weights[2] = (Math.round(weights[2] * 100.0) / 100.0);
		return output;
		// return "Source file :" + fileName + "\n Learning rate: " +
		// learningRate + "\n Total # weight updates = "
		// + counter + "\n Final weights:\n" + "\n" + "\n" + "\n";
	}
}
