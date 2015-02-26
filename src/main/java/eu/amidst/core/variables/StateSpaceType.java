package eu.amidst.core.variables;

import java.io.Serializable;

/**
 * Created by andresmasegosa on 25/11/14.
 */
public abstract class StateSpaceType implements Serializable {

    private static final long serialVersionUID = 4158293895929418259L;

    private StateSpaceTypeEnum stateSpaceTypeEnum;
    private String unit="NA";

    // This empty constructor is required because this class is the first non-serializable superclass in the inheritence
    // hierarchy for the classes FiniteStateSpace and RealStateSpace (both implements Serializable)
    public StateSpaceType(){}

    public StateSpaceType(StateSpaceTypeEnum type){
        this.stateSpaceTypeEnum =type;
    }

    public StateSpaceTypeEnum getStateSpaceTypeEnum(){
        return this.stateSpaceTypeEnum;
    }

    public String getUnit() {
        return unit;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }
}
