package eu.amidst.rlink;

import org.rosuda.REngine.REXPMismatchException;
import org.rosuda.REngine.REngineException;
import org.rosuda.REngine.Rserve.RConnection;
import org.rosuda.REngine.Rserve.RserveException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Created by rcabanas on 23/06/2017.
 */
public class PlotSeries {

	private List<double[]> Xvalues;
	private List<double[]> Yvalues;

	private RConnection c;

	private String plotParams;

	private String outputFormat ;

	private List<String> format;

	private List<Integer> pch, lty;

	private List<String> type, col;

	private Random random;

	private boolean addHashCode = false;





	public PlotSeries(RConnection c) throws RserveException {
		Xvalues = new ArrayList<double[]>();
		Yvalues = new ArrayList<double[]>();

		format = new ArrayList<String>();
		pch = new ArrayList<Integer>();
		lty = new ArrayList<Integer>();
		type = new ArrayList<String>();
		col = new ArrayList<String>();

		plotParams = "";

		if(c==null)
			this.c = RConnectionBuilder.getInstance();

		random = new Random(555);



	}

	public PlotSeries() throws RserveException {
		this(null);
	}



	public PlotSeries addSeries(double[] X, double[] Y) {

		this.Xvalues.add(X);
		this.Yvalues.add(Y);
		this.format.add("");

		return this;
	}

	public PlotSeries addSeries(double[] Y) {
		return addSeries(null, Y);
	}






	private void assignData() throws REngineException {
		for(int i=0; i<Yvalues.size(); i++){
			if(Xvalues.get(i) != null)
				c.assign("X"+i+""+getHashCode(), Xvalues.get(i));
			c.assign("Y"+i+""+getHashCode(), Yvalues.get(i));

		}
	}

	public String getAsignCode() throws REngineException {

		String rcmd = "";

		for(int i=0; i<Yvalues.size(); i++){
			if(Xvalues.get(i) != null) {

				rcmd += "X" + i + "" + getHashCode()+" <- c(" +
						Arrays.stream(Xvalues.get(i))
								.mapToObj(d -> d+"")
								.collect(Collectors.joining(", "))
						+ ")\n";
			}
			rcmd += "Y" + i + "" + getHashCode()+" <- c(" +
					Arrays.stream(Yvalues.get(i))
							.mapToObj(d -> d+"")
							.collect(Collectors.joining(", "))
					+ ")\n";

		}


		rcmd +="\n Y = cbind(";
		for(int i=0; i<Yvalues.size()-1; i++){
			rcmd += "Y"+i+", ";
		}
		rcmd += "Y"+(Yvalues.size()-1)+") ";
		rcmd +="\n";



/*		String outRcmd = "";
		int count = 0;
		for (char c: rcmd.toCharArray()) {
			if(c == ','){
				count++;
				if(count%50 == 0)
					outRcmd +=",\n";
				else
					outRcmd += ",";
			}else {
				outRcmd += c;
			}
		}


*/

		String outRcmd = "";

		int i = rcmd.indexOf(',');
		int count = 1;

		while(i!= -1) {

			System.out.println(count);
			String s = rcmd.substring(0,i+1);
			rcmd = rcmd.substring(i+1);

			if(count%50 == 0) {
				s = s+"\n";
			}

			outRcmd += s;
			count++;
			i = rcmd.indexOf(',');

		}


		outRcmd += rcmd;







		return outRcmd;
	}




	private void clearData() throws REngineException {
		for(int i=0; i<Yvalues.size(); i++){
			if(Xvalues.get(i) != null)
				c.eval("rm(X"+i+""+getHashCode()+")");
			c.eval("rm(Y"+i+""+getHashCode()+")");

		}
	}


	private PlotSeries setFormat(int indx, String format) {
		this.format.set(indx, format);
		return this;

	}


	public PlotSeries setFormat(int indx, int pch, String type, int lty, String col){

		String strFormat = "pch="+pch+", type='"+type+"', lty="+lty+", col='"+col+"'";

		this.pch.add(indx,pch);
		this.type.add(indx,type);
		this.lty.add(indx,lty);
		this.col.add(col);

		return this.setFormat(indx, strFormat);


	}


	private void save(String filename) throws REXPMismatchException, REngineException {


		assignData();

		String rcmd = "";

		rcmd += outputFormat+"(file=\""+filename+"\");\n";

		rcmd += getPlotCode();

		rcmd +="dev.off();\n";

		System.out.println(rcmd);

		c.parseAndEval(rcmd);

		clearData();

	}


	private String getPlotCode() {
		String rcmd = "";
		//rcmd += "title(main='main title', sub='sub-title', xlab='x-axis label', ylab='y-axis label'); \n";

		for(int i=0; i<Yvalues.size(); i++) {

			String xplot = "";
			if(Xvalues.get(i) != null)
				xplot = "X" + i + "" + getHashCode()+", ";

			if(i==0) {
				rcmd += "plot(" +xplot+" Y" + i + "" + getHashCode() + ", " + getFormat(i) + ","
						+plotParams+
						");\n";

			}else
				rcmd += "lines("+xplot+" Y"+i+""+getHashCode()+", "+getFormat(i)+");\n";

		}


		String lblStr = "c("+IntStream.range(0, Yvalues.size())
							.mapToObj(i -> "'"+String.valueOf(i)+"'")
							.collect(Collectors.joining(", "))+")";

		String colStr = "c("+col.stream().map(s -> "'"+s+"'" ).collect(Collectors.joining(", "))+")";
		String ltyStr = "c("+lty.stream().map(l -> l.toString()).collect(Collectors.joining(", "))+")";
		String pchStr = "c("+pch.stream().map(l -> l.toString()).collect(Collectors.joining(", "))+")";


		rcmd += "legend('topright', "+lblStr+", col="+colStr+", lty="+ltyStr+", pch="+pchStr+");\n";
		return rcmd;

	}

	private String getFormat(int indx) {
		if(format.get(indx) != "")
			return format.get(indx);

		//int[] markers = {"o", "l", "o"};
		int[] markers = {0,1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,19,20};
		int[] lines = {1,2,3,4,5,6};
		String[] colors = {"red", "red4", "orange", "sienna1", "black", "gray75", "darkgreen", "blue", "deepskyblue4", "darkturquoise" };


		int l = lines[random.nextInt(lines.length)];

		String c = colors[random.nextInt(colors.length)];

		int m = markers[random.nextInt(markers.length)];


		setFormat(indx, m, "b", l, c);

		return  "pch="+m
				+", type='b'"
				+ ", lty="+l
				+", col='"+c+"'";



	}






	public void toPDF(String filename) throws REXPMismatchException, REngineException {
		this.outputFormat = "pdf";
		save(filename);

	}

	private String getHashCode(){
		String str = "";
		if(isAddHashCode())
			str += "_"+hashCode();

		return str;

	}

	//Getters and setters

	public PlotSeries setSeed(int seed) {
		random = new Random(seed);
		return this;
	}

	public PlotSeries setPlotParams(String plotParams) {
		this.plotParams = plotParams;
		return this;
	}

	public PlotSeries setAddHashCode(boolean addHashCode) {
		this.addHashCode = addHashCode;
		return this;
	}

	public boolean isAddHashCode() {
		return addHashCode;
	}
	// testing method

	public static void main(String[] args) throws REngineException, REXPMismatchException {

		double[] Y0 = {3.0, 2.0, 1.0, 6.0, 8.9, 10, 10, 11, 3,4,4,4,5};
		double[] Y1 = {4.0, 4.0, 2.0, 5.0, 5.9, 8, 8, 11, 6,};
		double[] Y2 = {8.0, 8.0, 7.0, 7.5, 7.9, 7, 6, 6, 7,8,8,8};

		double[] X0 = {0,1,2,3,4,5,6,7,8,9,10,11,12};



		PlotSeries plotSeries = new PlotSeries()//.setSeed(1111)
					.setPlotParams("main='main title', xlab='x-axis label', ylab='y-axis label'")
						.addSeries(X0,Y0).setFormat(0, 1, "b", 4, "green")
						.addSeries(Y1)//.setFormat(1, 1, "b", 4, "sienna1")
						.addSeries(Y2);//.setFormat(2, 1, "b", 4, "green");


		System.out.println(plotSeries.getAsignCode());

		plotSeries.toPDF("./plotRserve.pdf");



	}



}
