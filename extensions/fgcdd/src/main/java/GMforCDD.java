import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.utils.DataSetGenerator;
import eu.amidst.latentvariablemodels.staticmodels.CustomGaussianMixture;
import eu.amidst.latentvariablemodels.staticmodels.GaussianMixture;
import eu.amidst.latentvariablemodels.staticmodels.Model;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;

/**
 * Created by rcabanas on 20/07/2017.
 */
public class GMforCDD {


	private static String filename = "./datasets/DriftSets/data_2var_nex_15k.arff";


	public static void main(String[] args) throws WrongConfigurationException {

		DataStream<DataInstance> data = DataStreamLoader.open(filename);


		GaussianMixture GMM = new GaussianMixture(data.getAttributes())
		.setDiagonal(true)
		.setNumStatesHiddenVar(2)
		.setWindowSize(1000);

		//GMM.updateModel(data);

		for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(1000)) {
			GMM.updateModel(batch);
		}
		System.out.println(GMM.getModel());
		System.out.println(GMM.getDAG());

		System.out.println("HiddenVar");
		System.out.println(GMM.getPosteriorDistribution("HiddenVar").toString());


	}



}
