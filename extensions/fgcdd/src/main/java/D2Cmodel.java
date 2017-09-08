import eu.amidst.core.datastream.*;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.learning.parametric.bayesian.BayesianParameterLearningAlgorithm;
import eu.amidst.core.learning.parametric.bayesian.SVB;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;
import eu.amidst.latentvariablemodels.staticmodels.classifiers.Classifier;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;
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
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The NaiveBayesClassifier class implements the interface {@link Classifier} and defines a Naive Bayes Classifier.
 * See Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press, page 82.
 */
public class D2Cmodel extends Classifier<D2Cmodel>{


	Variable hiddenVar;


	/**
	 * Constructor of the classifier which is initialized with the default arguments:
	 * the last variable in attributes is the class variable and importance sampling
	 * is the inference algorithm for making the predictions.
	 * @param attributes list of attributes of the classifier (i.e. its variables)
	 * @throws WrongConfigurationException is thrown when the attributes passed are not suitable
	 * for such classifier
	 */
	public D2Cmodel(Attributes attributes) throws WrongConfigurationException {
		super(attributes);
	}

	/**
	 * Builds the DAG over the set of variables given with the naive Bayes structure
	 */
	@Override
	protected void buildDAG() {




		HashMap<Variable, List<Variable>> contVars = new HashMap<Variable, List<Variable>>();
		//List<List<Variable>> contVars = new ArrayList<List<Variable>>();

		List<Variable> discVars = vars.getListOfVariables().stream()
									.filter(variable -> variable != classVar)
									.collect(Collectors.toList());


		for (Variable D: discVars) {
			List<Variable> Cset = new ArrayList<Variable>();

			for(int i=0; i<D.getNumberOfStates(); i++) {
				Variable CDi = vars.newGaussianVariable("C" + D.getName() + "_" + i);
				Cset.add(CDi);
			}

			contVars.put(D,Cset);
		}

		hiddenVar = vars.newGaussianVariable("HiddenVar");

		dag = new DAG(vars);


		for (Variable D: discVars) {

			dag.getParentSet(D).addParent(classVar);

			List<Variable> Cset = contVars.get(D);

			for (Variable CDi: Cset) {
				dag.getParentSet(CDi).addParent(D);
				//dag.getParentSet(D).addParent(CDi);
			}

			//contVars.get(D).forEach(v -> dag.getParentSet(v).addParent(D));


		}


		dag.getParentSet(
			contVars.get(vars.getVariableByName("kdddata.V1")).get(0)
		).addParent(hiddenVar);

		dag.getParentSet(
				contVars.get(vars.getVariableByName("kdddata.V1")).get(1)
		).addParent(hiddenVar);


		/*

		dag.getParentSet(hiddenVar).addParent(
			contVars.get(vars.getVariableByName("kdddata.V1")).get(0)
		);

		dag.getParentSet(hiddenVar).addParent(
				contVars.get(vars.getVariableByName("kdddata.V1")).get(1)
		);

			*/

	}






	//////////// example of use

	public static void main(String[] args) throws WrongConfigurationException {



		//DataStream<DataInstance> data = DataSetGenerator.generate(1234,500, 2, 3);

		DataStream<DataInstance> data = DataStreamLoader.open("./datasets/DriftSets/kddcup_V1_25K_1.arff");


		System.out.println(data.getAttributes().toString());

		String classVarName = "kdddata.V41";

		D2Cmodel model = new D2Cmodel(data.getAttributes())
				.setClassName(classVarName);
		
		model.updateModel(data);

		System.out.println(model);




	}
}




