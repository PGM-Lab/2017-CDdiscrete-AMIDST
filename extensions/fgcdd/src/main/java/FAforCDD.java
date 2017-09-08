import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.utils.DataSetGenerator;
import eu.amidst.latentvariablemodels.staticmodels.FactorAnalysis;
import eu.amidst.latentvariablemodels.staticmodels.Model;
import eu.amidst.latentvariablemodels.staticmodels.exceptions.WrongConfigurationException;

/**
 * Created by rcabanas on 20/07/2017.
 */
public class FAforCDD {

	private static String filename = "./datasets/DriftSets/data_2var_ex_15k.arff";


	public static void main(String[] args) throws WrongConfigurationException {

		DataStream<DataInstance> data = DataStreamLoader.open(filename);


		Model model = new FactorAnalysis(data.getAttributes())
				.setNumberOfLatentVariables(1);

		System.out.println(model.getDAG());

		model.updateModel(data);

//        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(1000)) {
//            model.updateModel(batch);
//        }

		System.out.println(model.getModel());

		System.out.println(model.getPosteriorDistribution("LatentVar0").toString());


	}


}
