package eu.amidst.core.inference.VMP_;

import eu.amidst.core.exponentialfamily.*;
import eu.amidst.core.utils.Utils;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.Variable;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by andresmasegosa on 03/02/15.
 */
public class Node {

    List<Node> parents;

    List<Node> children;

    Assignment assignment;

    EF_UnivariateDistribution QDist;

    EF_ConditionalDistribution PDist;

    boolean observed=false;

    SufficientStatistics sufficientStatistics;

    boolean isDone = false;

    boolean active = true;



    public Node(EF_ConditionalDistribution PDist) {
        this.PDist = PDist;
        this.QDist= this.PDist.getNewBaseEFUnivariateDistribution().randomInitialization(new Random(1));
        this.parents = new ArrayList<>();
        this.children = new ArrayList<>();
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public List<Node> getChildren() {
        return children;
    }

    public void setChildren(List<Node> children) {
        this.children = children;
    }

    public List<Node> getParents() {
        return parents;
    }

    public void setParents(List<Node> parents) {
        this.parents = parents;
    }

    public Assignment getAssignment() {
        return assignment;
    }

    public boolean isObserved() {
        return observed;
    }

    public void setAssignment(Assignment assignment) {
        this.assignment = assignment;
        if (!Utils.isMissingValue(this.assignment.getValue(this.getMainVariable()))){
            this.observed=true;
            sufficientStatistics = this.QDist.getSufficientStatistics(assignment);
        }
    }

    public EF_UnivariateDistribution getQDist() {
        return (isObserved())? null: QDist;
    }

    public MomentParameters getQMomentParameters(){
        return (isObserved())? (MomentParameters) this.sufficientStatistics: QDist.getMomentParameters();
    }

    public EF_ConditionalDistribution getPDist() {
        return PDist;
    }

    public Variable getMainVariable(){
        return this.PDist.getVariable();
    }

    public Stream<Message<NaturalParameters>> computeMessages(){


        Map<Variable, MomentParameters> momentParents = new HashMap<>();

        this.parents.stream().forEach(p -> momentParents.put(p.getMainVariable(), p.getQMomentParameters()));

        momentParents.put(this.getMainVariable(), this.getQMomentParameters());

        List<Message<NaturalParameters>> messages = this.parents.stream()
                                                                .filter(node -> node.isActive())
                                                                .filter(parent -> !parent.isObserved())
                                                                .map(parent -> this.newMessageToParent(parent, momentParents))
                                                                .collect(Collectors.toList());

        if (!isObserved() && isActive()) {
            messages.add(this.newSelfMessage(momentParents));
        }

        return messages.stream();
    }

    public Message<NaturalParameters> newMessageToParent(Node parent, Map<Variable, MomentParameters> momentChildCoParents){
        Message<NaturalParameters> message = new Message<>(parent.getMainVariable());
        message.setVector(this.PDist.getExpectedNaturalToParent(parent.getMainVariable(), momentChildCoParents));
        message.setDone(this.messageDoneToParent(parent.getMainVariable()));

        return message;
    }

    public Message<NaturalParameters> newSelfMessage(Map<Variable, MomentParameters> momentParents) {
        Message<NaturalParameters> message = new Message(this.getMainVariable());
        message.setVector(this.PDist.getExpectedNaturalFromParents(momentParents));
        message.setDone(this.messageDoneFromParents());

        return message;
    }

    private boolean messageDoneToParent(Variable parent){

        if (!this.isObserved())
            return false;

        for (Node node : this.getParents()){
            if (node.getMainVariable()!=parent && !node.isObserved())
                return false;
        }

        return true;
    }

    private boolean messageDoneFromParents(){

        for (Node node : this.getParents()){
            if (!node.isObserved())
                return false;
        }

        return true;
    }

    public void updateCombinedMessage(Message<NaturalParameters> message){
        this.QDist.setNaturalParameters(message.getVector());
        this.isDone = message.isDone();
    }

    public boolean isDone(){
        return isDone || this.observed;
    }
    public double computeELBO(){
        Map<Variable, MomentParameters> momentParents = new HashMap<>();

        this.parents.stream().forEach(p -> momentParents.put(p.getMainVariable(), p.getQMomentParameters()));

        momentParents.put(this.getMainVariable(), this.getQMomentParameters());

        double elbo=0;
        NaturalParameters expectedNatural = this.PDist.getExpectedNaturalFromParents(momentParents);

        if (!isObserved()) {
            expectedNatural.substract(this.QDist.getNaturalParameters());
            elbo += expectedNatural.dotProduct(this.QDist.getMomentParameters());
            elbo -= this.PDist.getExpectedLogNormalizer(momentParents);
            elbo += this.QDist.computeLogNormalizer();
        }else {
            elbo += expectedNatural.dotProduct(this.sufficientStatistics);
            elbo -= this.PDist.getExpectedLogNormalizer(momentParents);
            elbo += this.PDist.computeLogBaseMeasure(this.assignment);
        }

        return  elbo;
    }
}
