import eu.amidst.rlink.RConnectionBuilder;
import org.rosuda.REngine.REXP;
import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;
import org.rosuda.REngine.Rserve.RConnection;


/**
 * Created by rcabanas on 21/06/2017.
 */
public class pruebas {
	public static void main(String[] args) throws REngineException, REXPMismatchException, InterruptedException {

		RConnectionBuilder.setDEBUG(true);
		RConnection c = RConnectionBuilder.getInstance();

		REXP out = c.eval("R.version.string");
		System.out.println(out.asString());


		double[] d= c.eval("rnorm(100)").asDoubles();


		//Arrays.asList(d).forEach(v -> System.out.println(v));

		double[] Y1 = {3.0, 2.0, 1.0, 6.0, 8.9};
		c.assign("Y1", Y1);

		double[] Y2 = {1.0, 1.0, 2.0, 9.0, 8.9};
		c.assign("Y2", Y2);
		//System.out.println(c.eval("print(Y1)").asString());


		c.eval("Y1=Y1*2");
		d = c.eval("Y1").asDoubles();

		for(int i = 0; i<d.length; i++){
			System.out.println(d[i]);
		}


		//c.eval("png(filename=\"plotRserve.png\"); plot(Y); dev.off()");


	//	try {
			String rcmd = " pdf(file=\"/Users/rcabanas/plotRserve.pdf\"); plot(Y1, type=\"o\", col=\"blue\"); dev.off();";
			System.out.println(c.parseAndEval(rcmd).asString());
	/*	}catch
				(org.rosuda.REngine.Rserve.RserveException e ) {
			System.out.println(e.getRequestErrorDescription());

		}catch (org.rosuda.REngine.REngineException e) {
			System.out.println(e.getMessage());
		}*/


		rcmd = "setwd(\""+System.getProperty("user.dir")+"\")";
		System.out.println(rcmd);
		c.eval(rcmd);

		System.out.println(
		c.eval("getwd()").asString());
	}

}
