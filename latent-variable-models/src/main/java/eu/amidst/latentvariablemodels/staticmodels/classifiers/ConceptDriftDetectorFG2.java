/*
 *
 *
 *    Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
 *    See the NOTICE file distributed with this work for additional information regarding copyright ownership.
 *    The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use
 *    this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under the License is
 *    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under the License.
 *
 *
 */

package eu.amidst.latentvariablemodels.staticmodels.classifiers;

import COM.hugin.HAPI.DataSet;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.ParallelMLMissingData;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.DataSetGenerator;
import eu.amidst.core.utils.Utils;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;

import java.util.List;
import java.util.stream.Collectors;

/**

 */
public class ConceptDriftDetectorFG2 extends Classifier<ConceptDriftDetectorFG2>{


    private Variable hiddenVar;
    private int numStatesHiddenVar;




    /**
     * Constructor of the classifier which is initialized with the default arguments:
     * the last variable in attributes is the class variable and importance sampling
     * is the inference algorithm for making the predictions.
     * @param attributes list of attributes of the classifier (i.e. its variables)
     * @throws WrongConfigurationException is thrown when the attributes passed are not suitable
     * for such classifier
     */
    public ConceptDriftDetectorFG2(Attributes attributes) throws WrongConfigurationException {
        super(attributes);
        numStatesHiddenVar = 2;

      //  this.setLearningAlgorithm(new ParallelMLMissingData());
    }

    /**
     * Builds the DAG over the set of variables given with the naive Bayes structure
     */
    @Override
    protected void buildDAG() {

       // hiddenVar = vars.newMultinomialVariable("HiddenVar",numStatesHiddenVar);

		hiddenVar = vars.newGaussianVariable("HiddenVar");


        dag = new DAG(vars);
        dag.getParentSets().stream()
				.filter(w -> !w.getMainVar().equals(classVar))
				.filter(w -> !w.getMainVar().equals(hiddenVar))
				.forEach(w -> {
					w.addParent(classVar);
				});

		vars.getListOfVariables().stream()
				.filter(v -> !v.equals(classVar) && !v.equals(hiddenVar))
				.forEach(v -> {
					//dag.getParentSet(v).addParent(hiddenVar);
					dag.getParentSet(hiddenVar).addParent(v);
				});

    }


    /**
     * tests if the attributes passed as an argument in the constructor are suitable for this classifier
     * @return boolean value with the result of the test.
     */
    @Override
    public boolean isValidConfiguration(){
        boolean isValid = true;

		if(true)
		return isValid;

        long numFinite = vars.getListOfVariables().stream()
                .filter( v -> v.getStateSpaceTypeEnum().equals(StateSpaceTypeEnum.FINITE_SET))
                .count();


        if(numFinite == 0) {
            isValid = false;
            String errorMsg = "It should contain at least 1 discrete variable and the rest shoud be real";
            this.setErrorMessage(errorMsg);

        }

        return  isValid;

    }


    /////// Getters and setters

    public ConceptDriftDetectorFG2 setNumStatesHiddenVar(int numStatesHiddenVar) {
        this.numStatesHiddenVar = numStatesHiddenVar;
        return this;
    }


    //////////// example of use

    public static void main(String[] args) throws WrongConfigurationException {



		//DataStream<DataInstance> data = DataStreamLoader.open("./datasets/DriftSets/finegrainCD.arff");
		DataStream<DataInstance> data = DataStreamLoader.open("./datasets/DriftSets/finegrainCD.arff");


		System.out.println(data.getAttributes().toString());

        String classVarName = "V2";

        ConceptDriftDetectorFG2 nb = new ConceptDriftDetectorFG2(data.getAttributes());
        nb.setClassName(classVarName);

        nb.updateModel(data);
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(100)) {

            nb.updateModel(batch);
        }
        System.out.println(nb.getModel());
        System.out.println(nb.getDAG());





    }
}




