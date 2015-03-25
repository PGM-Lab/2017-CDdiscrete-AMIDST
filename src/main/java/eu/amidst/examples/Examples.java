package eu.amidst.examples;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.datastream.DynamicDataInstance;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Multinomial_MultinomialParents;
import eu.amidst.core.distribution.Normal_MultinomialParents;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.io.DynamicDataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.models.DynamicBayesianNetwork;
import eu.amidst.core.models.DynamicDAG;
import eu.amidst.core.variables.*;

import java.util.Arrays;

/**
 * Created by Hanen on 05/03/15.
 */
public class Examples {

    public static void BNExample() throws Exception{

        DataStream<DataInstance> data = DataStreamLoader.loadFromFile("datasets/staticData.arff");

        StaticVariables variables = new StaticVariables(data.getAttributes());

        Variable A = variables.getVariableByName("A");
        Variable B = variables.getVariableByName("B");
        Variable C = variables.getVariableByName("C");
        Variable D = variables.getVariableByName("D");

        Variable H = variables.newMultionomialVariable("H", Arrays.asList("TRUE", "FALSE"));

        DAG dag = new DAG(variables);

        dag.getParentSet(A).addParent(H);
        dag.getParentSet(B).addParent(H);
        dag.getParentSet(C).addParent(H);
        dag.getParentSet(D).addParent(H);

        System.out.println(dag.toString());

        BayesianNetwork bnet = BayesianNetwork.newBayesianNetwork(dag);
        System.out.println(bnet.toString());

        //BayesianNetworkSampler sampler = new BayesianNetworkSampler(network);
        //sampler.setSeed(0);
        //sampler.setParallelMode(true);

        //DataStream<DataInstance> dataStream = sampler.sampleToDataBase(10000);

        //ARFFDataWriter.writeToARFFFile(dataStream, "datasets/staticData2.arff");


        Multinomial_MultinomialParents distA = bnet.getDistribution(A);
        Assignment parentConf = new HashMapAssignment(H.getNumberOfStates());
        parentConf.setValue(H, 0);
        distA.getMultinomial(parentConf).setProbabilities(new double[]{0.7, 0.3});
        parentConf.setValue(H, 1);
        distA.getMultinomial(parentConf).setProbabilities(new double[]{0.2, 0.8});

        Normal_MultinomialParents distC = bnet.getDistribution(C);
        parentConf.setValue(H, 0);
        distC.getNormal(0).setMean(0.15);
        distC.getNormal(0).setVariance(0.25);
        parentConf.setValue(H, 1);
        distC.getNormal(1).setMean(0.24);
        distC.getNormal(1).setMean(1);

        System.out.println(bnet.toString());

    }


    public static void DBNExample() throws Exception {

        DataStream<DynamicDataInstance> data = DynamicDataStreamLoader.loadFromFile("datasets/dynamicData.arff");

        DynamicVariables dynamicVariables = new DynamicVariables(data.getAttributes());

        Variable A = dynamicVariables.getVariable("A");
        Variable B = dynamicVariables.getVariable("B");
        Variable C = dynamicVariables.getVariable("C");
        Variable D = dynamicVariables.getVariable("D");



        Variable H1 = dynamicVariables.newMultinomialDynamicVariable("H1",Arrays.asList("TRUE", "FALSE"));
        Variable H2 = dynamicVariables.newMultinomialDynamicVariable("H2", Arrays.asList("TRUE", "FALSE"));



        // Time 0: Parents at time 0 are automatically created when adding parents at time t !!!
        // Time t


        DynamicDAG dynamicDAG = new DynamicDAG(dynamicVariables);

        dynamicDAG.getParentSetTimeT(B).addParent(H1);
        dynamicDAG.getParentSetTimeT(C).addParent(H1);
        dynamicDAG.getParentSetTimeT(D).addParent(H1);
        dynamicDAG.getParentSetTimeT(B).addParent(H2);
        dynamicDAG.getParentSetTimeT(C).addParent(H2);
        dynamicDAG.getParentSetTimeT(D).addParent(H2);
        dynamicDAG.getParentSetTimeT(A).addParent(A.getInterfaceVariable());
        dynamicDAG.getParentSetTimeT(H1).addParent(H1.getInterfaceVariable());
        dynamicDAG.getParentSetTimeT(H2).addParent(H2.getInterfaceVariable());

        dynamicDAG.getParentSetTime0(B).addParent(A);
        dynamicDAG.getParentSetTime0(B).removeParent(H1);

        System.out.println(dynamicDAG.toString());

        DynamicBayesianNetwork dynamicbnet = DynamicBayesianNetwork.newDynamicBayesianNetwork(dynamicDAG);

        System.out.println(dynamicbnet.toString());


        Multinomial distA = dynamicbnet.getDistributionTime0(A);
        distA.setProbabilities(new double[]{0.1, 0.9});

        Normal_MultinomialParents distC = dynamicbnet.getDistributionTime0(C);

        Assignment parentConf = new HashMapAssignment(H1.getNumberOfStates()*H2.getNumberOfStates());

        parentConf.setValue(H1, 0);
        parentConf.setValue(H2, 0);
        distC.getNormal(parentConf).setMean(0.7);
        distC.getNormal(parentConf).setVariance(0.04);

        parentConf.setValue(H1, 0);
        parentConf.setValue(H2, 1);
        distC.getNormal(parentConf).setMean(0.4);
        distC.getNormal(parentConf).setVariance(1);

        parentConf.setValue(H1, 1);
        parentConf.setValue(H2, 0);
        distC.getNormal(parentConf).setMean(0.75);
        distC.getNormal(parentConf).setVariance(0.0025);

        parentConf.setValue(H1, 1);
        parentConf.setValue(H2, 1);
        distC.getNormal(parentConf).setMean(0.66);
        distC.getNormal(parentConf).setVariance(0.0016);

        System.out.println(dynamicbnet.toString());

        //DynamicBayesianNetworkSampler sampler = new DynamicBayesianNetworkSampler(dynamicbnet);
        //sampler.setSeed(0);
        //sampler.setParallelMode(true);
        //DataStream<DynamicDataInstance> dataStream = sampler.sampleToDataBase(1000,10);
        //ARFFDataWriter.writeToARFFFile(dataStream, "./datasets/dynamicData.arff");
    }


    public static void main(String[] args) throws Exception {
       // Examples.BNExample();
        Examples.DBNExample();
    }
}
