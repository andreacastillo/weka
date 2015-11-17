package weka;

//Andrea Castillo
import java.io.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

public class Perceptron extends Classifier implements weka.core.OptionHandler {

	double learningRate;
	String fileName;
	int numInstances = 0, EPOCH = 0, counter = 0, bias = 1;
	Instances data;
	double weights[] = new double[3];
	double x[], y[];
	int actual[];
	boolean classified;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		this.data = data;
		weights[0] = -0.6;
		weights[1] = 0.75;
		weights[2] = 0.5;
		x = new double[data.numInstances()];
		y = new double[data.numInstances()];
		actual = new int[data.numInstances()];
		char tempChar;
		double sum;
		int predicted, i, j;

		for (i = 0; i < EPOCH; i++) {
			System.out.println();
			String currentInstance[] = new String[3];
			System.out.print("Iteration: " + i + " ");

			for (j = 0; j < data.numInstances(); j++) {
				currentInstance = data.instance(j).toString().split(",");
				x[j] = Double.parseDouble(currentInstance[0]);
				y[j] = Double.parseDouble(currentInstance[1]);
				tempChar = currentInstance[2].toCharArray()[0];
				if (tempChar == 'a')
					actual[j] = 1;
				if (tempChar == 'b')
					actual[j] = -1;

				classified = false;
				sum = weights[0] * bias + weights[1] * x[j] + weights[2] * y[j];
				predicted = sum >= 0 ? 1 : -1;
				if (actual[j] == 1 && predicted == -1) {
					// System.out.println("actual[j] == -1 && predicted == 1");
					weights[0] = weights[0] + (2 * learningRate * bias);
					weights[1] = weights[1] + (2 * learningRate * x[j]);
					weights[2] = weights[2] + (2 * learningRate * y[j]);
				} else if (actual[j] == -1 && predicted == 1) {
					weights[0] = weights[0] - (2 * learningRate * bias);
					weights[1] = weights[1] - (2 * learningRate * x[j]);
					weights[2] = weights[2] - (2 * learningRate * y[j]);
					// System.out.println("actual[j] == 1 && predicted == -1
				} else if (actual[j] == predicted) {
					classified = true;
				}
				distributionForInstance(data.instance(j));
			}
		}

	}

	public double[] distributionForInstance(Instance instance) {
		double[] result = new double[2];
		if (classified) {
			result[0] = 1;
			result[1] = 0;
			System.out.printf("1");
		} else {
			result[0] = 0;
			result[1] = 1;
			System.out.printf("0");
		}
		return result;

	}

	public void setOptions(String[] options) throws IOException, Exception {
		BufferedReader datafile = SimpleWeka.readDataFile("simple.arff");
		fileName = "simple.arff";
		EPOCH = 10;// (Integer.parseInt(options[3]));
		learningRate = 0.2;// (Double.parseDouble(options[5]));
	}

	public String toString() {
		return " weight " + " " + weights[0] + " " + weights[1] + " " + weights[2];
		// return "Source file :" + fileName + "\n Learning rate: " +
		// learningRate + "\n Total # weight updates = "
		// + counter + "\n Final weights:\n" + weights[0] + "\n" +
		// weights[1] + "\n" + weights[2] + "\n";
	}
}
