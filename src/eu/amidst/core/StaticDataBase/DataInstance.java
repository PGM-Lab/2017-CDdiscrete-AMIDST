package eu.amidst.core.StaticDataBase;

/**
 * Created by afa on 02/07/14.
 */
public interface DataInstance {
    public double getValue(int varID);

    public void setValue(int varID, double value);

    public DataStream getDataStream();
}
