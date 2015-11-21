package weka;

//Andrea Castillo
//Kevin Anderson
//COP 4630
//Program 3
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Perceptron extends Classifier implements weka.core.OptionHandler {

	private static final long serialVersionUID = 1L;
	String fileName;
	int EPOCH = 0, index = 0;
	double bias = 1.0, actual, predicted, learningRate;
	Instances data;
	boolean classified;
	double[] weights;
	int[] values;
	

	@Override
	public void buildClassifier(Instances data) throws Exception {
		this.data = data;
		double sum = 0;
		int x = 0;
		weights = new double[data.numAttributes()];
		values = new int[EPOCH];
		for(int i = 0; i < weights.length; i++){
			weights[i] = Math.random();
		}

		for(int i = 0; i < EPOCH; i++){
			values[i] = 1;
		}
		for (int i = 0; i < EPOCH; i++) {
			System.out.print("Iteration " + i + ": ");
			for (int j = 0; j < data.numInstances(); j++) {
				for (x = 0; x < data.numAttributes(); x++) {
					if (x == 0)
						sum = (weights[0] * bias);
					else
						sum += (weights[x] * data.instance(j).value(x - 1));
				}
				// A is 0 B is 1 I am changing it to A is 1 and B is -1
				actual = data.instance(j).value(x - 1) == 0.0 ? 1.0 : -1.0;
				predicted = sum >= 0.0 ? 1.0 : -1.0;

				if (predicted == actual) {
					classified = true;
					System.out.printf("1");
					
				} else {
					classified = false;
					System.out.printf("0");
						values[i] = 0;
					
					// added actual to the equation. it changes the plus or
					// minus
					for (int m = 0; m < weights.length; m++) {
						if (m == 0)
							weights[0] = weights[0]
									+ (actual * 2 * learningRate * bias);
						else
							weights[m] = weights[m]
									+ (actual * 2 * learningRate * data
											.instance(j).value(m - 1));
					}
				}
			}
			System.out.println();

		}
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] out = new double[2];
		double sum = 0.0;
		for (int j = 0; j < weights.length; j++) {
			if (j == 0)
				sum += weights[j] * 1.0;
			else
				sum += weights[j] * instance.value(j - 1);
		}
		double actual = instance.value(instance.numAttributes() - 1);
		if (instance.attribute(instance.numAttributes() - 1).isNumeric()) {
			if (instance.value(instance.numAttributes() - 1) > 0)
				actual = 1.0;
			else
				actual = 0.0;
		} else {
			actual = instance.value(instance.numAttributes() - 1);
			if (actual == 1.0)
				actual = 0.0;
			else
				actual = 1.0;
		}
		if ((sum >= 0 && actual == 0.0) || (sum < 0 && actual == 1.0)) {
			out[0] = 0;
			out[1] = 1;
		} else {
			out[0] = 1;
			out[1] = 0;
		}
		return out;
	}

	public void setOptions(String[] options) throws IOException, Exception {
		fileName = options[1];
		EPOCH = (Integer.parseInt(options[3]));
		learningRate = (Double.parseDouble(options[5]));
	}

	public String toString() {
		String output = "Final weights:\n";
		for (int i = 0; i < weights.length; i++) {
			output += (Math.round(weights[i] * 100.0) / 100.0) + "\n";
		}
		return output;
	}
}
