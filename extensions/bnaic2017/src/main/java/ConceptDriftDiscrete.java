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
import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuIIDReplication;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.latentvariablemodels.staticmodels.Model;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;
import org.apache.commons.lang3.NotImplementedException;

import java.util.ArrayList;
import java.util.List;

/**

 */
public class ConceptDriftDiscrete extends Model<ConceptDriftDiscrete> {



    /** Represents the variance added when making a transition*/
    double transitionVariance;

    /** Represents the index of the class variable of the classifier*/
    int classIndex;


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
    public ConceptDriftDiscrete(Attributes attributes) throws WrongConfigurationException {
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
		throw new NotImplementedException("Concept drift discrete not implemented in Flink");

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

    public ConceptDriftDiscrete setTransitionVariance(double transitionVariance) {
        this.transitionVariance = transitionVariance;
        resetModel();
		return this;
    }

    public int getClassIndex() {
        return classIndex;
    }

    public ConceptDriftDiscrete setClassIndex(int classIndex) {
        this.classIndex = classIndex;
        resetModel();
		return this;
    }

    public int getSeed() {
        return seed;
    }

    public ConceptDriftDiscrete setSeed(int seed) {
        this.seed = seed;
        resetModel();
		return this;
    }

    public double getFading() {
        return fading;
    }

    public ConceptDriftDiscrete setFading(double fading) {
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

	public ConceptDriftDiscrete setNumHidden(int numHidden) {
		this.numHidden = numHidden;
		return this;
	}


	public boolean isLinksFromClass() {
		return linksFromClass;
	}

	public ConceptDriftDiscrete setLinksFromClass(boolean linksFromClass) {
		this.linksFromClass = linksFromClass;
		return this;
	}

	public List<int[]> getMapIndx() {
		return mapIndx;
	}

	public ConceptDriftDiscrete setMapIndx(List<int[]> mapIndx) {
		this.mapIndx = mapIndx;
		return this;
	}


	public boolean isClassPresent() {
		return classPresent;
	}

	public ConceptDriftDiscrete setClassPresent(boolean classPresent) {
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


}

