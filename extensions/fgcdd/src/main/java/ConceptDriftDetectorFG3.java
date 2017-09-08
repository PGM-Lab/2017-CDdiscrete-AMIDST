/*
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * See the License for the specific language governing permissions and limitations under the License.
 *
 */

import eu.amidst.core.conceptdrift.utils.GaussianHiddenTransitionMethod;
import eu.amidst.core.datastream.*;
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
import java.util.OptionalInt;

/**

 */
public class ConceptDriftDetectorFG3 extends Model<ConceptDriftDetectorFG3> {

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

	private Variable classVariable;

	private boolean linksFromClass = true;
	private boolean classPresent = true;

	List<int[]>  mapIndx = null;


    /**
     * Constructor of classifier from a list of attributes (e.g. from a datastream).
     * The following parameters are set to their default values: numStatesHiddenVar = 2
     * and diagonal = true.
     * @param attributes object of the class Attributes
     */
    public ConceptDriftDetectorFG3(Attributes attributes) throws WrongConfigurationException {
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





        String className = "";

		if(isClassPresent()) {
			className = atts.getFullListOfAttributes().get(classIndex).getName();
			classVariable = vars.getVariableByName(className);
		}


		hiddenVars = new ArrayList<Variable>();

		for (int i = 0; i < numHidden ; i++) {
			hiddenVars.add(vars.newGaussianVariable("LocalHidden_"+i));
		}



		dag = new DAG(vars);



		List<List<Variable>> mapHidden = getMapHiddenVars();


		int j = 0;
        for (Attribute att : atts.getListOfNonSpecialAttributes()) {
            if (isClassPresent() && att.getName().equals(className))
                continue;

            Variable variable = vars.getVariableByName(att.getName());

			if(isLinksFromClass() && isClassPresent())
            	dag.getParentSet(variable).addParent(classVariable);

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





    /////// Getters and setters

    public double getTransitionVariance() {
        return transitionVariance;
    }

    public ConceptDriftDetectorFG3 setTransitionVariance(double transitionVariance) {
        this.transitionVariance = transitionVariance;
        resetModel();
		return this;
    }

    public int getClassIndex() {
        return classIndex;
    }

    public ConceptDriftDetectorFG3 setClassIndex(int classIndex) {
        this.classIndex = classIndex;
        resetModel();
		return this;
    }

    public int getSeed() {
        return seed;
    }

    public ConceptDriftDetectorFG3 setSeed(int seed) {
        this.seed = seed;
        resetModel();
		return this;
    }

    public double getFading() {
        return fading;
    }

    public ConceptDriftDetectorFG3 setFading(double fading) {
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

	public ConceptDriftDetectorFG3 setNumHidden(int numHidden) {
		this.numHidden = numHidden;
		return this;
	}


	public boolean isLinksFromClass() {
		return linksFromClass;
	}

	public ConceptDriftDetectorFG3 setLinksFromClass(boolean linksFromClass) {
		this.linksFromClass = linksFromClass;
		return this;
	}

	public List<int[]> getMapIndx() {
		return mapIndx;
	}

	public ConceptDriftDetectorFG3 setMapIndx(List<int[]> mapIndx) {
		this.mapIndx = mapIndx;
		return this;
	}


	public boolean isClassPresent() {
		return classPresent;
	}

	public  ConceptDriftDetectorFG3 setClassPresent(boolean classPresent) {
		this.classPresent = classPresent;
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
				if (att.getName().equals(classVariable.getName()))
					continue;

				List<Variable> Hset = new ArrayList<Variable>();
				Hset.add(hiddenVars.get(j));
				map.add(Hset);


				j++;

			}
		}
		return map;

	}








	//////////// example of use

//	private static String filename = "./datasets/DriftSets/kddcup_d2cn_V1_25k_1.arff";
private static String filename = "./datasets/DriftSets/syntheticNoClass_2k_1.arff";
	private static int windowSize = 1000;

	public static void main(String[] args) throws Exception {


		//computeNaiveBayes();
		//if(true)return;


		DataStream<DataInstance> data = DataStreamLoader.open(filename);

		System.out.println(data.getAttributes().toString());

		List<int[]> map = Arrays.asList(

				new int[][]{
						new int[]{0},
						new int[]{1},
						new int[]{0,1},
						new int[]{},
						new int[]{1},
						new int[]{1},
						new int[]{},
				}

		/*		new int[][]{
						new int[]{},
						new int[]{0},
						new int[]{},
						new int[]{},
						new int[]{0},
						new int[]{0},
						new int[]{},
				}*/

	/*
				new int[][]{
						new int[]{0},
						new int[]{},
						new int[]{0},
						new int[]{},
						new int[]{},
						new int[]{},
						new int[]{},
				}
	*/

		);


		int numHidden = map.stream().mapToInt(
				v -> {
					OptionalInt max = Arrays.stream(v).max();

					if(max.isPresent())
						return max.getAsInt();
					return 0;
				}).max().getAsInt() + 1;
				//map.stream().mapToInt(v -> Arrays.stream(v).min().getAsInt()).min().getAsInt();

	//	int numHidden = 2;

		System.out.println(numHidden);
		//Build the model
		Model model =
				new ConceptDriftDetectorFG3(data.getAttributes())
						.setWindowSize(windowSize)
						//.setClassIndex(3)
						.setTransitionVariance(0.1)
						.setNumHidden(numHidden)
						.setMapIndx(map)
						.setLinksFromClass(false)
						.setClassPresent(false);





		List<List<Double>> series = new ArrayList<List<Double>>();
	//	int numHidden = ((ConceptDriftDetectorFG3)model).getNumHidden();


		for(int i = 0; i<numHidden; i++) {
			series.add(new ArrayList<Double>());
		}


		int Nbatches = 0;
		int nfiles = 6;


		for(int f=1; f<=nfiles; f++) {

			if (f > 1) {
				filename = filename.replace((f - 1) + ".arff", f + ".arff");
				data = DataStreamLoader.open(filename);
			}

			System.out.println(filename);

			for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(windowSize)) {
				model.updateModel(batch);


				double[] localHidenMeans = ((ConceptDriftDetectorFG3) model).getLocalHidenMeans();
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

