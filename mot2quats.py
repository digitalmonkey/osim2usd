# For OpenSim support
import math, os, sys, getopt, numpy
import opensim as osim
import csv

def saveOutputCSV(outputPath, times, outputNames, outputData):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []

    for i in range(len(times)):
        row = []

        if i == 0: # Write header
            header.append("time")
            for label in outputNames:
                header.append(label)
            writer.writerow(header)

        row.append(str(times[i]))
        for data in outputData[i]:
            row.append(str(data))
        writer.writerow(row)

    file.close()
def saveMotionCSV(outputPath, bodyNames, times, poseTrajectories):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []
    for i in range(len(times)):
        row = []  # Row is a list of strings

        if i == 0:
            header.append("time")
        row.append(str(times[i]))
        bodyPoses = poseTrajectories[i]
        for body in range(len(bodyPoses)):
            if i == 0: # Add header info
                header.append(bodyNames[body] + "_x")
                header.append(bodyNames[body] + "_y")
                header.append(bodyNames[body] + "_z")
                header.append(bodyNames[body] + "_qw")
                header.append(bodyNames[body] + "_qx")
                header.append(bodyNames[body] + "_qy")
                header.append(bodyNames[body] + "_qz")

            # Add position
            row.append(str(bodyPoses[body][0][0]))
            row.append(str(bodyPoses[body][0][1]))
            row.append(str(bodyPoses[body][0][2]))

            # Add rotation as quaternion
            # Have to use get() method since can't seem to use subscript to asVec4.
            row.append(str(bodyPoses[body][1].get(0)))
            row.append(str(bodyPoses[body][1].get(1)))
            row.append(str(bodyPoses[body][1].get(2)))
            row.append(str(bodyPoses[body][1].get(3)))

        if i == 0: # Write header before 1st timestamp data
            writer.writerow(header)
        writer.writerow(row)
    file.close()
def mot2quats(motionPath, outputPath, modelPath, optionsDict):

    print(f"Input OpenSim motion path set to: {motionPath}")

    model = osim.Model(modelPath)
    model.initSystem()

    bodyNames = []
    for body in model.getBodyList():
        bodyNames.append(body.getName())

    bodySet = model.getBodySet()
    motion = osim.Storage(motionPath)
    motion.setInDegrees(False)

    # Results  states in motionTrajectory are in SimTK:Stage:Instance
    motionTrajectory = osim.StatesTrajectory.createFromStatesStorage(model, motion, True, True)
    print("Trajectory size: ", motionTrajectory.getSize(), " is compatible: ", motionTrajectory.isCompatibleWith(model))

    poseTrajectories = []
    times = []
    for i in range(motionTrajectory.getSize()):
        motionState = motionTrajectory.get(i)
        model.realizePosition(motionState)
        times.append(motionState.getTime())

        # Loop through bodies in model.
        bodyPoses = []
        for body in model.getBodyList():
            positionGround = body.getPositionInGround(motionState)
            rotationGround = body.getRotationInGround(motionState).convertRotationToQuaternion()
            bodyPoses.append((positionGround, rotationGround))
        poseTrajectories.append(bodyPoses)

    if outputPath != None:
        saveMotionCSV(outputPath + "_motion.csv", bodyNames, times, poseTrajectories)

        outputNames = []
        for force in model.getForceSet():
            outputNames.append(force.getName())

        outputData = []

        # Filter out activations only.
        names = model.getStateVariableNames()
        activationLabels = []
        for j in range(names.getSize()):
            if "/activation" in names.get(j):
                activationLabels.append(names.get(j))

        print(f"Found {len(activationLabels)} activation state variables for output.")

        outputNames = []
        for i in range(len(times)):
            sample = []
            state = motionTrajectory.get(i)
            model.realizeVelocity(state)

            for label in activationLabels:
                if i == 0:
                   muscleLabel = label.replace("/activation","")
                   muscleLabel = muscleLabel.replace("/forceset/","")
                   outputNames.append(muscleLabel)
                activation = model.getStateVariableValue(state, label)
                sample.append(str(activation))
            outputData.append(sample)

        saveOutputCSV(outputPath + "_output.csv", times, outputNames, outputData)



    return (bodyNames, times, poseTrajectories)


def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    motionPath = "./Motions/kinematics_activations_left_leg_squat_0.mot"
    inputPath = sessionPath + motionPath
    outputPath = os.path.splitext(inputPath)[0]
    optionsDict = dict()

    opts, args = getopt.getopt(argv,"im:o:",["input=","model=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("mot2quats.py -i <inputFile> -o <outputFile> -m <modelFile>")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputPath = arg
        elif opt in ("-o", "--output"):
            outputPath = arg
        elif opt in ("-m", "--model"):
            modelPath = arg

    bodyPoseTrajectories = mot2quats(inputPath, outputPath, modelPath, optionsDict)

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])