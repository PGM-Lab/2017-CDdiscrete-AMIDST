import eu.amidst.core.conceptdrift.utils.GaussianHiddenTransitionMethod;
import eu.amidst.core.datastream.*;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.inference.InferenceEngine;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuIIDReplication;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.flinklink.core.conceptdrift.IdentifiableIDAModel;
import eu.amidst.flinklink.core.learning.parametric.ParallelVB;
import eu.amidst.latentvariablemodels.staticmodels.Model;
import eu.amidst.latentvariablemodels.staticmodels.classifiers.NaiveBayesClassifier;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;
import eu.amidst.rlink.PlotSeries;
import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**

 */
public class ConceptDriftDetector_Hd_noclass extends Model<ConceptDriftDetector_Hd_noclass> {

    /** Represents the drift detection mode. Only the global mode is currently provided.*/
    public enum DriftDetector {GLOBAL};

    /** Represents the variance added when making a transition*/
    double transitionVariance;

    /** Represents the index of the class variable of the classifier*/
    int classIndex;

    /** Represents the drift detection mode. Only the global mode is currently provided.*/
    DriftDetector conceptDriftDetector;

    /** Represents the seed of the class*/
    int seed;


    /** Represents the list of hidden vars modelling concept drift*/
    List<Variable> hiddenVars;

	int numHidden;

    /** Represents the fading factor.*/
    double fading;


	List<int[]>  mapIndx = null;


    /**
     * Constructor of classifier from a list of attributes (e.g. from a datastream).
     * The following parameters are set to their default values: numStatesHiddenVar = 2
     * and diagonal = true.
     * @param attributes object of the class Attributes
     */
    public ConceptDriftDetector_Hd_noclass(Attributes attributes) throws WrongConfigurationException {
        super(attributes);

        transitionVariance=0.1;
        classIndex = atts.getNumberOfAttributes()-1;
        seed = 0;
        fading = 1.0;
		super.windowSize = 1000;
		numHidden = atts.getNumberOfAttributes()-1;
    }




    /**
     * Builds the DAG over the set of variables given with the structure of the model
     */
    @Override
    protected void buildDAG() {


        String className = atts.getFullListOfAttributes().get(classIndex).getName();
        hiddenVars = new ArrayList<Variable>();

		for (int i = 0; i < numHidden ; i++) {
			hiddenVars.add(vars.newMultinomialVariable("LocalHidden_"+i, 2));
		}



		dag = new DAG(vars);



		List<List<Variable>> mapHidden = getMapHiddenVars();


		int j = 0;
        for (Attribute att : atts.getListOfNonSpecialAttributes()) {

            Variable variable = vars.getVariableByName(att.getName());

			for (Variable v: mapHidden.get(j)) {
				dag.getParentSet(variable).addParent(v);
			}



			j++;

        }



    }

    @Override
   protected  void initLearning() {

        if (this.getDAG()==null)
            buildDAG();

        if(learningAlgorithm==null) {
            SVB svb = new SVB();
            svb.setSeed(this.seed);
            svb.setPlateuStructure(new PlateuIIDReplication(hiddenVars));
            GaussianHiddenTransitionMethod gaussianHiddenTransitionMethod = new GaussianHiddenTransitionMethod(hiddenVars, 0, this.transitionVariance);
            gaussianHiddenTransitionMethod.setFading(fading);
            svb.setTransitionMethod(gaussianHiddenTransitionMethod);
            svb.setDAG(dag);

            svb.setOutput(true);
            svb.getPlateuStructure().getVMP().setMaxIter(1000);
            svb.getPlateuStructure().getVMP().setThreshold(0.001);

			//svb.setParallelMode(true);

            learningAlgorithm = svb;
        }
        learningAlgorithm.setWindowsSize(windowSize);
        if (this.getDAG()!=null)
            learningAlgorithm.setDAG(this.getDAG());
        else
            throw new IllegalArgumentException("Non provided dag");

        learningAlgorithm.setOutput(false);
        learningAlgorithm.initLearning();
        initialized=true;
    }


	protected void initLearningFlink() {

		if (this.getDAG()==null)
			buildDAG();



		if(learningAlgorithmFlink==null) {

			ParallelVB svb = new ParallelVB();
			svb.setSeed(this.seed);
			svb.setPlateuStructure(new PlateuIIDReplication(hiddenVars));
			GaussianHiddenTransitionMethod gaussianHiddenTransitionMethod = new GaussianHiddenTransitionMethod(hiddenVars, 0, this.transitionVariance);
			gaussianHiddenTransitionMethod.setFading(1.0);
			svb.setTransitionMethod(gaussianHiddenTransitionMethod);
			svb.setBatchSize(this.windowSize);
			svb.setDAG(dag);
			svb.setIdenitifableModelling(new IdentifiableIDAModel());

			svb.setOutput(false);
			svb.setMaximumGlobalIterations(100);
			svb.setMaximumLocalIterations(100);
			svb.setGlobalThreshold(0.001);
			svb.setLocalThreshold(0.001);

			learningAlgorithmFlink = svb;

		}

		learningAlgorithmFlink.setBatchSize(windowSize);

		if (this.getDAG()!=null)
			learningAlgorithmFlink.setDAG(this.getDAG());
		else
			throw new IllegalArgumentException("Non provided dag");


		learningAlgorithmFlink.initLearning();
		initialized=true;

		System.out.println("Window Size = "+windowSize);
	}



	public double[] getLocalHidenMeans() {

		double means[] = new double[hiddenVars.size()];

		for (int i = 0; i<means.length; i++) {
			means[i] = ((Normal) this.getPosteriorDistribution(hiddenVars.get(i).getName())).getMean();

		}

		return means;

	}

	public double[] getLocalHidenProb0() {

		double means[] = new double[hiddenVars.size()];

		for (int i = 0; i<means.length; i++) {
			means[i] = ((Multinomial) this.getPosteriorDistribution(hiddenVars.get(i).getName())).getProbabilityOfState(0);

		}

		return means;

	}



    /////// Getters and setters

    public double getTransitionVariance() {
        return transitionVariance;
    }

    public ConceptDriftDetector_Hd_noclass setTransitionVariance(double transitionVariance) {
        this.transitionVariance = transitionVariance;
        resetModel();
		return this;
    }

    public int getClassIndex() {
        return classIndex;
    }

    public ConceptDriftDetector_Hd_noclass setClassIndex(int classIndex) {
        this.classIndex = classIndex;
        resetModel();
		return this;
    }

    public int getSeed() {
        return seed;
    }

    public ConceptDriftDetector_Hd_noclass setSeed(int seed) {
        this.seed = seed;
        resetModel();
		return this;
    }

    public double getFading() {
        return fading;
    }

    public ConceptDriftDetector_Hd_noclass setFading(double fading) {
        this.fading = fading;
        resetModel();
		return this;
    }

	public List<Variable> getHiddenVars() {
		return hiddenVars;
	}


	public int getNumHidden() {
		return numHidden;
	}

	public ConceptDriftDetector_Hd_noclass setNumHidden(int numHidden) {
		this.numHidden = numHidden;
		return this;
	}

	public List<int[]> getMapIndx() {
		return mapIndx;
	}

	public ConceptDriftDetector_Hd_noclass setMapIndx(List<int[]> mapIndx) {
		this.mapIndx = mapIndx;
		return this;
	}


	private List<List<Variable>> createMapHiddenFromIndexSets(){
		List<List<Variable>> map = new ArrayList<List<Variable>>();

		for (int [] Hindx : mapIndx) {
			List<Variable> Hset = new ArrayList<Variable>();
			for (int i: Hindx) {
				Hset.add(this.hiddenVars.get(i));
			}

			map.add(Hset);
		}

		return map;
	}





	private List<List<Variable>> getMapHiddenVars(){

		List<List<Variable>> map = null;


		if(mapIndx != null) {
			map = createMapHiddenFromIndexSets();
		}
		else{
			map = new ArrayList<List<Variable>>();


			int j = 0;
			for (Attribute att : atts.getListOfNonSpecialAttributes()) {

				List<Variable> Hset = new ArrayList<Variable>();
				Hset.add(hiddenVars.get(j));
				map.add(Hset);


				j++;

			}
		}
		return map;

	}





	public static void computeNaiveBayes() throws REngineException, REXPMismatchException {

	/*	String filename = "./datasets/DriftSets/kddcup_V1_25k_1.arff";

		DataStream<DataInstance> data = DataStreamLoader.open(filename);

		int nfiles = 20;

		boolean update = true;


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

				System.out.println(model.getModel());

				int i = 0;
				for (Variable v : model.getDAG().getVariables().getListOfVariables()) {
					if (v != model.getClassVar()) {
						double[] p = InferenceEngine.getPosterior(v, model.getModel()).getParameters();
						//	Doubles.asList(p).forEach(d -> System.out.print(d));
						//	System.out.println();
						for (int j = 0; j < p.length; j++) {
							series.get(i).add(p[j]);
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

*/

	}



	//////////// example of use

	private static String filename = "./datasets/DriftSets/data_1var_15k.arff";
	private static int windowSize = 1000;

	public static void main(String[] args) throws Exception {


		//computeNaiveBayes();
		//if(true)return;


		DataStream<DataInstance> data = DataStreamLoader.open(filename);

		System.out.println(data.getAttributes().toString());

		List<int[]> map = Arrays.asList(
				new int[][]{
						new int[]{0},
					//	new int[]{1},
					//	new int[]{0},
				}
		);


		int numHidden = map.stream().mapToInt(v -> Arrays.stream(v).max().getAsInt()).max().getAsInt() + 1;
				//map.stream().mapToInt(v -> Arrays.stream(v).min().getAsInt()).min().getAsInt();

		System.out.println(numHidden);
		//Build the model
		Model model =
				new ConceptDriftDetector_Hd_noclass(data.getAttributes())
						.setWindowSize(windowSize)
						.setTransitionVariance(0.1)
						.setNumHidden(numHidden)
						.setMapIndx(map);









		List<List<Double>> series = new ArrayList<List<Double>>();


		for(int i = 0; i<numHidden; i++) {
			series.add(new ArrayList<Double>());
		}


		int Nbatches = 0;
		int nfiles = 20;


		for(int f=1; f<=nfiles; f++) {

			if (f > 1) {
				filename = filename.replace((f - 1) + ".arff", f + ".arff");
				data = DataStreamLoader.open(filename);
			}

			System.out.println(filename);

			for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(windowSize)) {
				model.updateModel(batch);


				double[] localHidenMeans = ((ConceptDriftDetector_Hd_noclass) model).getLocalHidenProb0();
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


		plotSeries.toPDF("./plotCD.pdf");



	}



}

