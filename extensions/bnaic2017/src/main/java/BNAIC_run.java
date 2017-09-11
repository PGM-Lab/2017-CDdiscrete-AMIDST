import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.inference.InferenceEngine;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.variables.Variable;
import eu.amidst.latentvariablemodels.staticmodels.Model;
import eu.amidst.latentvariablemodels.staticmodels.classifiers.NaiveBayesClassifier;
import eu.amidst.rlink.PlotSeries;
import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by rcabanas on 23/08/2017.
 */
public class BNAIC_run {






	public static void main(String[] args) throws Exception {



		//// Synthetic - Concept Drift detection ////

		Properties synthetic = new Properties();
		synthetic.classPresent = false;
		synthetic.linksFromClass = false;
		synthetic.windowSize = 1000;
		synthetic.filename = "./datasets/DriftSets/syntheticNoClass_2k_1.arff";
		synthetic.nfiles = 6;

		synthetic.map = Arrays.asList(
				new int[][]{
						new int[]{0},
						new int[]{1},
						new int[]{0},
						new int[]{},
						new int[]{1},
						new int[]{1},
						new int[]{},
				}
		);








		/// Synthetic - Naive Bayes Learning ////


		Properties syntheticNB = new Properties();
		syntheticNB.classPresent = true;
		syntheticNB.linksFromClass = true;
		syntheticNB.windowSize = 1000;
		syntheticNB.filename = "./datasets/DriftSets/syntheticDiscrete.arff";
		syntheticNB.nfiles = 1;
		syntheticNB.update = false;


		//// intrussion - Concept Drift detection ////

		Properties intrussion = new Properties();
		intrussion.classPresent = false;
		intrussion.linksFromClass = false;
		intrussion.windowSize = 1000;
		intrussion.filename = "./datasets/DriftSets/kddcupNoClass_25k_1.arff";
		intrussion.nfiles = 20;


		intrussion.map = IntStream.range(0,80) // 0,79
						.mapToObj( i ->  new int[]{} )
						.collect(Collectors.toList());




		intrussion.map.set(0, new int[]{0});
		intrussion.map.set(2, new int[]{0});
		intrussion.map.set(3, new int[]{0});


		intrussion.map.set(71, new int[]{1});
		intrussion.map.set(75, new int[]{1});





		/// intrussion - Naive Bayes Learning ////


		Properties intrussionNB = new Properties();
		intrussionNB.classPresent = true;
		intrussionNB.linksFromClass = true;
		intrussionNB.windowSize = 1000;
		intrussionNB.filename = "./datasets/DriftSets/kddcup_discrete.arff";
		intrussionNB.nfiles = 1;
		intrussionNB.update = true;





		/// Invoke experiments

		run(synthetic);
		run(intrussion);

		// probabilities computation
		computeNaiveBayes(syntheticNB);
		computeNaiveBayes(intrussionNB);




	}




	private static void run(Properties p) throws REngineException {


		/// Experiment configuration ///

		String filename = p.filename;
		int windowSize = p.windowSize;
		boolean classPresent = p.classPresent;
		boolean linksFromClass = p.linksFromClass;
		int nfiles = p.nfiles;



		List<int[]> map = p.map;



		// Run experiment

		DataStream<DataInstance> data = DataStreamLoader.open(filename);

		System.out.println(data.getAttributes().toString());


		int numHidden = map.stream().mapToInt(
				v -> {
					OptionalInt max = Arrays.stream(v).max();

					if(max.isPresent())
						return max.getAsInt();
					return 0;
				}).max().getAsInt() + 1;


		System.out.println(numHidden);

		//Build the model
		Model model =
				new ConceptDriftDiscrete(data.getAttributes())
						.setWindowSize(windowSize)
						//.setClassIndex(3)
						.setTransitionVariance(0.1)
						.setNumHidden(numHidden)
						.setMapIndx(map)
						.setLinksFromClass(linksFromClass)
						.setClassPresent(classPresent);





		List<List<Double>> series = new ArrayList<List<Double>>();
		//	int numHidden = ((ConceptDriftDiscrete)model).getNumHidden();


		for(int i = 0; i<numHidden; i++) {
			series.add(new ArrayList<Double>());
		}


		int Nbatches = 0;

		for(int f=1; f<=nfiles; f++) {

			if (f > 1) {
				filename = filename.replace((f - 1) + ".arff", f + ".arff");
				data = DataStreamLoader.open(filename);
			}

			System.out.println(filename);

			for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(windowSize)) {
				model.updateModel(batch);


				double[] localHidenMeans = ((ConceptDriftDiscrete) model).getLocalHidenMeans();
				System.out.println(Arrays.toString(localHidenMeans)
						.replace("[", "")
						.replace("]", "")
						.replace(",", "\t"));


				for (int i = 0; i < numHidden; i++) {
					series.get(i).add(localHidenMeans[i]);
				}

				Nbatches++;
				System.out.println("batch" + Nbatches);

			}

		}


		//Plot....

		PlotSeries plotSeries = new PlotSeries()
				.setPlotParams("main='main title', xlab='x-axis label', ylab='y-axis label'");


		for(int i = 0; i<numHidden; i++) {

			double[] Yi = series.get(i).stream().mapToDouble(D -> D).toArray();

			plotSeries.addSeries(Yi);
		}


		System.out.println(plotSeries.getAsignCode());



	}



	public static void computeNaiveBayes(Properties p) throws REngineException, REXPMismatchException {

		String filename = p.filename;
		int nfiles = p.nfiles;
		boolean update = p.update;
		int windowSize = p.windowSize;



		DataStream<DataInstance> data = DataStreamLoader.open(filename);





		System.out.println(data.getAttributes().toString());

		int classIndex = data.getAttributes().getNumberOfAttributes() - 1;
		String className = data.getAttributes().getFullListOfAttributes().get(classIndex).getName();

		NaiveBayesClassifier model;

		List<List<Double>> series = new ArrayList<List<Double>>();


		int numSeries =
				data.getAttributes().getFullListOfAttributes()
						.stream()
						.mapToInt(a -> {
							int d = 0;
							if(a.getName() != className){
								d = a.getNumberOfStates();
							}
							System.out.println(a.getName()+" -> "+d);
							return d;
						}).sum();

		for(int i = 0; i<numSeries; i++) {
			series.add(new ArrayList<Double>());
		}
		int Nbatches = 0;


		model = null;

		if(update)
			model = new NaiveBayesClassifier(data.getAttributes())
					.setClassName(className)
					.setWindowSize(windowSize);


		for(int f=1; f<=nfiles; f++) {

			if(f>1) {
				filename = filename.replace((f-1)+".arff", f+".arff");
				data = DataStreamLoader.open(filename);
			}

			System.out.println(filename);


			for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(windowSize)) {

				if (!update)
					model = new NaiveBayesClassifier(data.getAttributes())
							.setClassName(className)
							.setWindowSize(windowSize);


				model.updateModel(batch);


				Nbatches++;
				System.out.println("batch " + Nbatches);

				System.out.print(".");
				//System.out.println(model.getModel());

				int i = 0;
				for (Variable v : model.getDAG().getVariables().getListOfVariables()) {
					if (v != model.getClassVar()) {
						double[] pr = InferenceEngine.getPosterior(v, model.getModel()).getParameters();
						//	Doubles.asList(p).forEach(d -> System.out.print(d));
						//	System.out.println();
						for (int j = 0; j < pr.length; j++) {
							series.get(i).add(pr[j]);
							i++;
						}

					}
				}
			}
		}

		PlotSeries plotSeries = new PlotSeries()
				.setPlotParams("main='main title', xlab='x-axis label', ylab='y-axis label'");


		for(int i = 0; i<numSeries; i++) {

			double[] Yi = series.get(i).stream().mapToDouble(D -> D).toArray();

			plotSeries.addSeries(Yi);
		}


		System.out.println(plotSeries.getAsignCode());


		plotSeries.toPDF("./plotNB.pdf");


		System.out.println(Nbatches+" batches");



	}




	private static class Properties{

		Properties() {

		}

		String filename = "./datasets/DriftSets/syntheticNoClass_2k_1.arff";
		int windowSize = 1000;
		boolean classPresent = true;
		boolean linksFromClass = true;
		int nfiles = 1;

		List<int[]> map = null;

		boolean update = true;



	}


}
