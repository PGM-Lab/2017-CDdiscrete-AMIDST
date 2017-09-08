package eu.amidst.rlink;

import org.rosuda.REngine.REXP;
import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.Rserve.RConnection;
import org.rosuda.REngine.Rserve.RserveException;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rcabanas on 22/06/2017.
 */
public class RConnectionBuilder {

	private static final int maxConnections = 1;

	private static final String initCmd = "Rscript -e library(Rserve) -e Rserve(args='--no-save')";


	private static List<RConnection> list = new ArrayList<RConnection>();


	private static boolean DEBUG = false;


	public static RConnection getInstance() throws RserveException {
		if(list.size()==0) {
			addConnection();
		}
		return list.get(0);
	}


	public static void stop() {
		if(list.size()>0) {
			list.get(0).close();
			list.remove(0);
		}

	}


	public static void reset() throws RserveException {
		stop();
		getInstance();

	}


	private static void addConnection() throws RserveException {

		String out = ExecuteShellCommand.run(initCmd);
		if(DEBUG)
			System.out.println(out);

		RConnection c = new RConnection();

		String rcmd = "setwd(\""+System.getProperty("user.dir")+"\")";
		c.eval(rcmd);

		list.add(c);

	}


	/// geters andd setters ///


	public static List<RConnection> getList() {
		return list;
	}

	public static int getMaxConnections() {
		return maxConnections;
	}

	public static void setDEBUG(boolean DEBUG) {
		RConnectionBuilder.DEBUG = DEBUG;
	}


	//////

	public static void main(String[] args) throws RserveException, REXPMismatchException {


		RConnectionBuilder.setDEBUG(true);
		RConnection c = RConnectionBuilder.getInstance();

		REXP out = c.eval("R.version.string");
		System.out.println(out.asString());


		RConnectionBuilder.reset();
		//c.close();

		RConnection c2 = RConnectionBuilder.getInstance();


	}


}
