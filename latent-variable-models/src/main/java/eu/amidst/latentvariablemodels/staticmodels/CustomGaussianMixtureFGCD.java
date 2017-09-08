package eu.amidst.latentvariablemodels.staticmodels;

import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.latentvariablemodels.staticmodels.classifiers.Classifier;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by rcabanas on 23/05/16.
 */

public class CustomGaussianMixtureFGCD extends Model<CustomGaussianMixtureFGCD>{

	Attributes attributes;

	Variable classVar;

	List<Variable> localHidden;

	public CustomGaussianMixtureFGCD(Attributes attributes) throws WrongConfigurationException {
		super(attributes);
		this.attributes=attributes;

	}

	@Override
	protected void buildDAG() {


		//Obtain the predictive attributes
		List<Variable> attrVars = vars.getListOfVariables().stream()
				.filter(v -> !v.equals(classVar))
				.collect(Collectors.toList());

		int numAttr = attrVars.size();

		/** Create a hidden variable with two hidden states */
	//	Variable globalHiddenVar = vars.newMultinomialVariable("globalHiddenVar",6);

		/** Create a list of local hidden variables */
		localHidden = new ArrayList<Variable>();
		//for(int i= 0; i< numAttr; i++) {
		for(int i= 0; i< 2; i++) {

		//	localHidden.add(vars.newMultinomialVariable("localHiddenVar_"+i,2));
			localHidden.add(vars.newGaussianVariable("localHiddenVar_"+i));

		}


		/** We create a standard naive Bayes */
		DAG dag = new DAG(vars);


/*		/** Add the links */
		for (int i=0; i<numAttr; i++) {
	//		dag.getParentSet(attrVars.get(i)).addParent(localHidden.get(i));
	//		dag.getParentSet(attrVars.get((i+1)%6)).addParent(localHidden.get(i));

		//	dag.getParentSet(attrVars.get(i)).addParent(globalHiddenVar);
			dag.getParentSet(attrVars.get(i)).addParent(classVar);
		//	dag.getParentSet(classVar).addParent(vars.getVariableByName("localHiddenVar_"+i));

		}


		dag.getParentSet(attrVars.get(0)).addParent(localHidden.get(0));
		dag.getParentSet(attrVars.get(1)).addParent(localHidden.get(0));
//		dag.getParentSet(classVar).addParent(localHidden.get(0));


		dag.getParentSet(attrVars.get(2)).addParent(localHidden.get(1));
		dag.getParentSet(attrVars.get(3)).addParent(localHidden.get(1));
//		dag.getParentSet(classVar).addParent(localHidden.get(0));


/*		dag.getParentSet(attrVars.get(4)).addParent(localHidden.get(1));
		dag.getParentSet(attrVars.get(5)).addParent(localHidden.get(1));
//		dag.getParentSet(classVar).addParent(localHidden.get(1));


		dag.getParentSet(attrVars.get(6)).addParent(localHidden.get(1));
		dag.getParentSet(attrVars.get(7)).addParent(localHidden.get(1));
//		dag.getParentSet(classVar).addParent(localHidden.get(1));


		dag.getParentSet(attrVars.get(8)).addParent(localHidden.get(2));
		dag.getParentSet(attrVars.get(9)).addParent(localHidden.get(2));
//		dag.getParentSet(classVar).addParent(localHidden.get(2));


		dag.getParentSet(attrVars.get(10)).addParent(localHidden.get(2));
		dag.getParentSet(attrVars.get(11)).addParent(localHidden.get(2));
//		dag.getParentSet(classVar).addParent(localHidden.get(2));

*/

		//This is needed to maintain coherence in the Model class.
		this.dag=dag;


	}

	public CustomGaussianMixtureFGCD setClassVar(Variable classVar) {
		this.classVar = classVar;
		return this;
	}

	public CustomGaussianMixtureFGCD setClassName(String className) {
		return this.setClassVar(vars.getVariableByName(className));
	}

	public List<Variable> getLocalHidden() {
		return localHidden;
	}

	//Method for testing the custom model
	public static void main(String[] args) {
		String filename = "datasets/bnaic2015/BCC/Month0.arff";

		int windowSize = 100;
		filename = "./datasets/DriftSets/finegrainCDbin4vars.arff";
		DataStream<DataInstance> data = DataStreamLoader.open(filename);




		data.streamOfBatches(windowSize).forEach(
				batch -> {

					Model model = new CustomGaussianMixtureFGCD(data.getAttributes())
							.setClassName("V5")
							.setWindowSize(windowSize);
					model.updateModel(batch);
					//System.out.println(model.getModel());

					for(int i=0; i<((CustomGaussianMixtureFGCD)model).getLocalHidden().size(); i++)
						System.out.print(((Normal)model.getPosteriorDistribution("localHiddenVar_"+i)).getMean()+"\t");
		//				System.out.print(((Multinomial) model.getPosteriorDistribution("localHiddenVar_" + i)).getProbabilityOfState(0)+"\t");
					//	System.out.print(((Multinomial) model.getPosteriorDistribution("globalHiddenVar")).getProbabilityOfState(i)+"\t");

		//			System.out.print(((Normal)model.getPosteriorDistribution("globalHiddenVar")).getMean()+"\t");


					System.out.println("");
			//		System.out.println(model.getModel());

				}
		);
		//System.out.println(model.getModel());


	}


}
